//+------------------------------------------------------------------+
//|                                            GoldMindCopier.mq5      |
//|                                                                    |
//| Copia sul conto MT5 (demo) i segnali che GoldMind (il bot Telegram |
//| su Railway) apre in paper trading. Ogni PollSeconds secondi legge  |
//| /api/ea/pending: per ogni ordine nuovo apre la posizione/pending   |
//| corrispondente con lo stesso entry/SL/TP, poi conferma con         |
//| /api/ea/ack cosi' il server non lo ripropone al giro successivo.   |
//|                                                                    |
//| Non modifica ne' chiude mai una POSIZIONE già aperta/riempita      |
//| (propria o altrui) — l'unica eccezione (dal 2026-09-18, bug reale  |
//| segnalato dall'utente: un BUY LIMIT cancellato lato bot restava    |
//| aperto su MT5) e' cancellare un ordine PENDING (mai riempito, MAI  |
//| una posizione) quando GoldMind lo invalida — vedi /api/ea/         |
//| cancellations sotto. GoldMind continua a fare la propria           |
//| simulazione paper esattamente come prima: questo EA e' un          |
//| consumatore in piu' dello stesso segnale, non sostituisce nulla.   |
//|                                                                    |
//| Lottaggio: se RiskBasedSizing=true (default) il lotto e' calcolato |
//| dal risk_pct del trade sul saldo REALE di questo conto (stesso     |
//| standard di risk_manager.calculate_lot_size() lato bot, coi tick   |
//| value veri del broker) — non un lotto fisso identico per ogni      |
//| segnale a prescindere dallo stop loss o dal capitale disponibile.  |
//+------------------------------------------------------------------+
#property copyright "GoldMind"
#property version   "1.00"

#include <Trade\Trade.mqh>
CTrade trade;

input string ServerUrl        = "https://goldmind-bot-production.up.railway.app"; // URL GoldMind (Railway)
input string ApiToken         = "";           // stesso DASHBOARD_TOKEN configurato su Railway
input int    PollSeconds      = 5;            // ogni quanto controllare nuovi segnali
input bool   RiskBasedSizing  = true;         // true: lotto calcolato da risk_pct sul saldo REALE del conto; false: usa sempre LotSize
input double LotSize          = 0.01;         // lotto fisso di riserva (usato se RiskBasedSizing=false o se il calcolo dinamico fallisce)
input double MaxLotSize       = 1.0;          // tetto di sicurezza: mai superato, qualunque cosa dica il calcolo dinamico
input string SymbolToTrade    = "XAUUSD+";    // simbolo oro su questo broker — verificare il nome ESATTO in Market Watch (alcuni broker usano suffissi come "+", ".m", "-ECN": un nome sbagliato fa restituire 0 a ogni SymbolInfo*, causando errori che sembrano di tutt'altro tipo — scadenza, tipo ordine, lotto — vedi diagnostica 2026-09-17)

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(990100);
   if(ApiToken == "")
      Print("ATTENZIONE: ApiToken vuoto — /api/ea/pending risponderà 401, nessun ordine verrà copiato.");
   EventSetTimer(PollSeconds);
   string sizingMode = RiskBasedSizing
      ? StringFormat("lotto dinamico da risk_pct (tetto %.2f, riserva %.2f se non calcolabile)", MaxLotSize, LotSize)
      : StringFormat("lotto fisso %.2f", LotSize);
   Print("GoldMindCopier avviato — polling ", ServerUrl, " ogni ", PollSeconds, "s su ", SymbolToTrade,
         " — ", sizingMode);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   CheckForNewSignals();
   CheckForCancellations();
}

//+------------------------------------------------------------------+
//| Legge /api/ea/pending, elabora ogni ordine nell'array JSON         |
//+------------------------------------------------------------------+
void CheckForNewSignals()
{
   string url = ServerUrl + "/api/ea/pending?token=" + ApiToken;
   char post[];
   char result[];
   string headers;

   ResetLastError();
   int res = WebRequest("GET", url, "", 5000, post, result, headers);
   if(res == -1)
   {
      int err = GetLastError();
      if(err == 4060)
         Print("WebRequest bloccata: aggiungi ", ServerUrl,
               " in Strumenti > Opzioni > Expert Advisor > 'Consenti WebRequest per gli URL elencati'");
      else
         Print("WebRequest fallita, errore ", err);
      return;
   }
   if(res != 200)
   {
      Print("Risposta HTTP ", res, " da /api/ea/pending — token sbagliato o server non raggiungibile?");
      return;
   }

   string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   ProcessOrdersJson(body);
}

//+------------------------------------------------------------------+
//| Spezza l'array JSON piatto "[{...},{...}]" nei singoli oggetti.    |
//| MQL5 non ha un parser JSON nativo: essendo lo schema fisso e       |
//| controllato solo da GoldMind (mai testo libero/annidato dentro     |
//| ogni oggetto), spezzare sulle graffe bilanciate ed estrarre i      |
//| campi per posizione di stringa e' sufficiente e robusto qui.       |
//+------------------------------------------------------------------+
void ProcessOrdersJson(string body)
{
   string trimmed = body;
   StringTrimLeft(trimmed);
   StringTrimRight(trimmed);
   if(StringLen(trimmed) < 2)
      return; // "[]" o risposta vuota

   int depth = 0;
   int objStart = -1;
   for(int i = 0; i < StringLen(trimmed); i++)
   {
      ushort ch = StringGetCharacter(trimmed, i);
      if(ch == '{')
      {
         if(depth == 0) objStart = i;
         depth++;
      }
      else if(ch == '}')
      {
         depth--;
         if(depth == 0 && objStart >= 0)
         {
            string obj = StringSubstr(trimmed, objStart, i - objStart + 1);
            ProcessOneOrder(obj);
            objStart = -1;
         }
      }
   }
}

//+------------------------------------------------------------------+
string JsonStringValue(string json, string key)
{
   string pattern = "\"" + key + "\":\"";
   int p = StringFind(json, pattern);
   if(p < 0) return "";
   p += StringLen(pattern);
   int e = StringFind(json, "\"", p);
   if(e < 0) return "";
   return StringSubstr(json, p, e - p);
}

double JsonNumberValue(string json, string key)
{
   string pattern = "\"" + key + "\":";
   int p = StringFind(json, pattern);
   if(p < 0) return 0.0;
   p += StringLen(pattern);
   int e = p;
   while(e < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, e);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-')
         e++;
      else
         break;
   }
   return StringToDouble(StringSubstr(json, p, e - p));
}

//+------------------------------------------------------------------+
//| Stesso standard di risk_manager.calculate_lot_size() lato bot, ma  |
//| coi tick value REALI di questo broker per SymbolToTrade invece di  |
//| assumere 100 oz/lotto — SYMBOL_TRADE_TICK_VALUE/TICK_SIZE danno il  |
//| valore monetario per 1 lotto di qualunque distanza di prezzo,      |
//| corretto anche se il contratto di XAUUSD+ differisse da quello     |
//| "standard" assunto lato Python (che dimensiona solo il saldo       |
//| VIRTUALE del paper trading, mai usato per il conto vero).          |
//+------------------------------------------------------------------+
double CalculateDynamicLot(double riskPct, double entry, double sl)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double slDistance = MathAbs(entry - sl);
   double tickValue = SymbolInfoDouble(SymbolToTrade, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(SymbolToTrade, SYMBOL_TRADE_TICK_SIZE);

   if(balance <= 0 || riskPct <= 0 || slDistance <= 0 || tickValue <= 0 || tickSize <= 0)
   {
      Print("Lotto dinamico non calcolabile (balance=", balance, " risk_pct=", riskPct,
            " sl_distance=", slDistance, ") — uso LotSize di riserva ", LotSize);
      return LotSize;
   }

   double valuePerLot = slDistance / tickSize * tickValue;
   double riskAmount  = balance * riskPct / 100.0;
   double rawLot      = riskAmount / valuePerLot;

   double lotStep = SymbolInfoDouble(SymbolToTrade, SYMBOL_VOLUME_STEP);
   double lotMin  = SymbolInfoDouble(SymbolToTrade, SYMBOL_VOLUME_MIN);
   double lotMax  = SymbolInfoDouble(SymbolToTrade, SYMBOL_VOLUME_MAX);
   if(lotStep <= 0) lotStep = 0.01;

   double steppedLot = MathFloor(rawLot / lotStep) * lotStep;
   steppedLot = MathMax(steppedLot, lotMin);
   steppedLot = MathMin(steppedLot, MathMin(lotMax, MaxLotSize));
   steppedLot = NormalizeDouble(steppedLot, 2);

   if(steppedLot < lotMin)
   {
      Print("Lotto dinamico (", DoubleToString(rawLot, 4), ") sotto il minimo broker (", lotMin,
            ") anche dopo l'arrotondamento — uso LotSize di riserva ", LotSize);
      return LotSize;
   }

   Print("Lotto dinamico: saldo=", DoubleToString(balance, 2), " risk=", riskPct,
         "% -> rischio $", DoubleToString(riskAmount, 2), " / SL ", DoubleToString(slDistance, 2),
         " -> lotto ", DoubleToString(steppedLot, 2));
   return steppedLot;
}

//+------------------------------------------------------------------+
//| Alcuni broker (visto in produzione il 2026-09-17 su Ultima Markets|
//| Demo) non supportano ORDER_TIME_GTC per gli ordini pending: CTrade|
//| lo rifiuta con "Unable to place order without explicitly         |
//| specified expiration time" / retcode "invalid expiration", e il   |
//| BUY/SELL LIMIT/STOP non viene mai piazzato. SYMBOL_EXPIRATION_MODE|
//| dice quali modalita' il simbolo accetta davvero su QUESTO broker: |
//| si usa GTC solo se supportato, altrimenti DAY, altrimenti un      |
//| orario esplicito (SPECIFIED) 30 giorni nel futuro — abbastanza    |
//| lungo da non scadere mai prima che il bot lato server cancelli il |
//| pending per conto proprio (vedi invalidazione pending in          |
//| trade_manager.py, molto più stretta).                             |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_TIME PickSupportedExpiration(datetime &expirationOut)
{
   long modes = SymbolInfoInteger(SymbolToTrade, SYMBOL_EXPIRATION_MODE);
   expirationOut = 0;
   if((modes & SYMBOL_EXPIRATION_GTC) != 0)
      return ORDER_TIME_GTC;
   if((modes & SYMBOL_EXPIRATION_DAY) != 0)
      return ORDER_TIME_DAY;
   expirationOut = TimeCurrent() + 30 * 24 * 60 * 60;
   if((modes & SYMBOL_EXPIRATION_SPECIFIED) != 0)
      return ORDER_TIME_SPECIFIED;
   if((modes & SYMBOL_EXPIRATION_SPECIFIED_DAY) != 0)
      return ORDER_TIME_SPECIFIED_DAY;
   // Bitmask a 0 o non riconosciuta: SPECIFIED con una scadenza reale
   // resta il tentativo più compatibile in assoluto anche quando il
   // broker non dichiara esplicitamente nessuna modalità.
   return ORDER_TIME_SPECIFIED;
}

//+------------------------------------------------------------------+
//| Diagnostica (2026-09-17): dopo aver risolto l'errore "invalid      |
//| expiration" con PickSupportedExpiration sopra, il BUY LIMIT ha     |
//| iniziato a fallire con un errore diverso ("CTrade::OrderTypeCheck: |
//| Invalid order type") — un secondo blocco più a valle, probabilmente|
//| legato a quali tipi di ordine/riempimento questo simbolo accetta   |
//| davvero su questo broker (SYMBOL_ORDER_MODE/SYMBOL_FILLING_MODE),  |
//| non più alla scadenza. Invece di tirare a indovinare una seconda   |
//| volta (e far ricompilare a vuoto), si stampa una volta per         |
//| tentativo fallito la diagnostica completa del simbolo — i valori   |
//| reali dicono con certezza cosa correggere.                         |
//+------------------------------------------------------------------+
void LogSymbolTradeCapabilities()
{
   Print("Diagnostica ", SymbolToTrade, ": SYMBOL_ORDER_MODE=", SymbolInfoInteger(SymbolToTrade, SYMBOL_ORDER_MODE),
         " SYMBOL_EXPIRATION_MODE=", SymbolInfoInteger(SymbolToTrade, SYMBOL_EXPIRATION_MODE),
         " SYMBOL_FILLING_MODE=", SymbolInfoInteger(SymbolToTrade, SYMBOL_FILLING_MODE),
         " SYMBOL_TRADE_MODE=", SymbolInfoInteger(SymbolToTrade, SYMBOL_TRADE_MODE));
}

//+------------------------------------------------------------------+
//| Apre l'ordine corrispondente a un oggetto JSON, poi conferma.      |
//| Dedup locale via GlobalVariable (a livello di terminale, non solo  |
//| di questo EA) oltre alla rimozione server-side via ack: se l'ack   |
//| di un giro precedente fosse fallito per un problema di rete,       |
//| questo evita comunque di riaprire due volte lo stesso ordine.      |
//+------------------------------------------------------------------+
void ProcessOneOrder(string obj)
{
   string tradeId   = JsonStringValue(obj, "trade_id");
   string orderType = JsonStringValue(obj, "order_type");
   double entry = JsonNumberValue(obj, "entry");
   double sl    = JsonNumberValue(obj, "sl");
   double tp1   = JsonNumberValue(obj, "tp1");
   double riskPct = JsonNumberValue(obj, "risk_pct");

   if(tradeId == "")
      return;

   string seenVar = "GM_seen_" + tradeId;
   if(GlobalVariableCheck(seenVar))
   {
      AckOrder(tradeId); // già aperto in un giro precedente — solo conferma di nuovo
      return;
   }

   string ot = orderType;
   StringToUpper(ot);
   bool sent = false;
   string cmt = "GoldMind " + tradeId;

   double lot = RiskBasedSizing ? CalculateDynamicLot(riskPct, entry, sl) : LotSize;

   datetime expiration;
   ENUM_ORDER_TYPE_TIME typeTime = PickSupportedExpiration(expiration);

   if(ot == "BUY")
      sent = trade.Buy(lot, SymbolToTrade, 0.0, sl, tp1, cmt);
   else if(ot == "SELL")
      sent = trade.Sell(lot, SymbolToTrade, 0.0, sl, tp1, cmt);
   else if(ot == "BUY LIMIT")
      sent = trade.BuyLimit(lot, entry, SymbolToTrade, sl, tp1, typeTime, expiration, cmt);
   else if(ot == "SELL LIMIT")
      sent = trade.SellLimit(lot, entry, SymbolToTrade, sl, tp1, typeTime, expiration, cmt);
   else if(ot == "BUY STOP")
      sent = trade.BuyStop(lot, entry, SymbolToTrade, sl, tp1, typeTime, expiration, cmt);
   else if(ot == "SELL STOP")
      sent = trade.SellStop(lot, entry, SymbolToTrade, sl, tp1, typeTime, expiration, cmt);
   else
   {
      Print("Tipo ordine sconosciuto: '", orderType, "' (trade_id ", tradeId, ") — scartato senza aprire nulla.");
      AckOrder(tradeId);
      return;
   }

   double fillPrice = 0.0;
   if(sent)
   {
      // ResultPrice() e' il prezzo di esecuzione REALE solo per un
      // ordine a mercato appena eseguito - per un LIMIT/STOP appena
      // piazzato (non ancora attivato) ritornerebbe il prezzo richiesto,
      // non un vero fill, quindi non lo inviamo (slippage tracking,
      // 2026-09-15 — solo ordini a mercato per ora).
      if(ot == "BUY" || ot == "SELL")
         fillPrice = trade.ResultPrice();
      Print("Copiato: ", ot, " ", SymbolToTrade, " lotto=", DoubleToString(lot, 2),
            " entry=", entry, " sl=", sl, " tp=", tp1,
            (fillPrice > 0 ? " fill=" + DoubleToString(fillPrice, 2) : ""), " (", tradeId, ")");
      GlobalVariableSet(seenVar, 1);
      AckOrder(tradeId, fillPrice);
   }
   else
   {
      // BUG REALE (2026-09-17): prima si chiamava AckOrder anche qui, in
      // ogni caso — un ordine che falliva ad aprirsi (es. il bug della
      // scadenza dei pending, vedi PickSupportedExpiration) veniva
      // comunque tolto per sempre dalla coda /api/ea/pending: il server
      // pensava fosse stato gestito, l'EA non lo riproponeva mai più al
      // giro successivo, e quel trade non arrivava MAI sul conto MT5,
      // silenziosamente. ack_broker_order() lato server documenta già
      // questo esattamente ("con successo o con un errore che non ha
      // senso ritentare") ma qui non veniva rispettato. Ora un fallimento
      // NON conferma nulla: l'ordine resta in coda e viene ritentato al
      // prossimo poll (PollSeconds) — innocuo se il problema persiste
      // (solo log ripetuti), decisivo se invece era un bug ormai corretto.
      Print("Errore apertura ", ot, " per ", tradeId, ": ", trade.ResultRetcodeDescription(),
            " — ordine lasciato in coda, verrà ritentato al prossimo giro.");
      LogSymbolTradeCapabilities();
   }
}

//+------------------------------------------------------------------+
void AckOrder(string tradeId, double fillPrice = 0.0)
{
   string url  = ServerUrl + "/api/ea/ack?token=" + ApiToken;
   string json = fillPrice > 0
      ? "{\"trade_id\":\"" + tradeId + "\",\"fill_price\":" + DoubleToString(fillPrice, 2) + "}"
      : "{\"trade_id\":\"" + tradeId + "\"}";

   char post[];
   int len = StringToCharArray(json, post) - 1; // esclude lo zero terminatore
   ArrayResize(post, len);

   char result[];
   string headers;
   ResetLastError();
   int res = WebRequest("POST", url, "Content-Type: application/json\r\n", 5000, post, result, headers);
   if(res == -1)
      Print("Ack fallito per ", tradeId, ": errore ", GetLastError());
}

//+------------------------------------------------------------------+
//| Cancellazioni (2026-09-18, bug reale segnalato dall'utente: un     |
//| BUY LIMIT cancellato lato bot restava aperto su MT5 — l'EA sopra   |
//| non aveva NESSUN modo di saperlo, essendo "solo apertura" per      |
//| design). Legge /api/ea/cancellations ogni PollSeconds secondi,     |
//| cerca tra gli ordini PENDING (mai una posizione già riempita, che  |
//| resta fuori scope) quello col commento "GoldMind <trade_id>" e lo  |
//| cancella con OrderDelete — stesso schema di CheckForNewSignals,    |
//| direzione opposta.                                                 |
//+------------------------------------------------------------------+
void CheckForCancellations()
{
   string url = ServerUrl + "/api/ea/cancellations?token=" + ApiToken;
   char post[];
   char result[];
   string headers;

   ResetLastError();
   int res = WebRequest("GET", url, "", 5000, post, result, headers);
   if(res == -1)
   {
      int err = GetLastError();
      if(err != 4060) // gia' segnalato da CheckForNewSignals allo stesso giro
         Print("WebRequest cancellazioni fallita, errore ", err);
      return;
   }
   if(res != 200)
      return; // gia' segnalato da CheckForNewSignals allo stesso giro

   string body = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   ProcessCancellationsJson(body);
}

//+------------------------------------------------------------------+
//| Stesso spezzamento sulle graffe bilanciate di ProcessOrdersJson,   |
//| duplicato invece di condiviso: MQL5 rende scomodo passare un       |
//| puntatore a funzione qui, e i due schemi JSON sono comunque         |
//| diversi (un oggetto con un solo campo contro molti) — vedi anche    |
//| JsonStringValue/JsonNumberValue sopra, stesso principio del resto  |
//| di questo file.                                                    |
//+------------------------------------------------------------------+
void ProcessCancellationsJson(string body)
{
   string trimmed = body;
   StringTrimLeft(trimmed);
   StringTrimRight(trimmed);
   if(StringLen(trimmed) < 2)
      return; // "[]" o risposta vuota

   int depth = 0;
   int objStart = -1;
   for(int i = 0; i < StringLen(trimmed); i++)
   {
      ushort ch = StringGetCharacter(trimmed, i);
      if(ch == '{')
      {
         if(depth == 0) objStart = i;
         depth++;
      }
      else if(ch == '}')
      {
         depth--;
         if(depth == 0 && objStart >= 0)
         {
            string obj = StringSubstr(trimmed, objStart, i - objStart + 1);
            ProcessOneCancellation(obj);
            objStart = -1;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Cerca un ordine PENDING (mai una posizione) col commento           |
//| "GoldMind <tradeId>" e lo cancella. Se non lo trova (mai piazzato,  |
//| gia' scaduto sul broker, o gia' cancellato in un giro precedente   |
//| con ack fallito) conferma comunque subito: non c'e' nulla da        |
//| ritentare. Se lo trova ma OrderDelete fallisce, NON conferma —      |
//| resta in coda e viene ritentato al prossimo giro, stesso principio |
//| di ProcessOneOrder per le aperture.                                 |
//+------------------------------------------------------------------+
void ProcessOneCancellation(string obj)
{
   string tradeId = JsonStringValue(obj, "trade_id");
   if(tradeId == "")
      return;

   string wantedComment = "GoldMind " + tradeId;
   ulong ticket = 0;
   for(int i = 0; i < OrdersTotal(); i++)
   {
      ulong t = OrderGetTicket(i);
      if(t > 0 && OrderGetString(ORDER_COMMENT) == wantedComment)
      {
         ticket = t;
         break;
      }
   }

   if(ticket == 0)
   {
      AckCancel(tradeId); // mai piazzato o gia' sparito — niente da cancellare
      return;
   }

   if(trade.OrderDelete(ticket))
   {
      Print("Pending cancellato su MT5: ticket=", ticket, " (", tradeId, ")");
      AckCancel(tradeId);
   }
   else
   {
      Print("Errore cancellazione ticket=", ticket, " per ", tradeId, ": ",
            trade.ResultRetcodeDescription(), " — resta in coda, verrà ritentato al prossimo giro.");
   }
}

//+------------------------------------------------------------------+
void AckCancel(string tradeId)
{
   string url  = ServerUrl + "/api/ea/ack-cancel?token=" + ApiToken;
   string json = "{\"trade_id\":\"" + tradeId + "\"}";

   char post[];
   int len = StringToCharArray(json, post) - 1;
   ArrayResize(post, len);

   char result[];
   string headers;
   ResetLastError();
   int res = WebRequest("POST", url, "Content-Type: application/json\r\n", 5000, post, result, headers);
   if(res == -1)
      Print("Ack-cancel fallito per ", tradeId, ": errore ", GetLastError());
}
