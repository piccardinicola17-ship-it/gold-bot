//+------------------------------------------------------------------+
//|                                            GoldMindCopier.mq5      |
//|                                                                    |
//| Copia sul conto MT5 (demo) i segnali che GoldMind (il bot Telegram |
//| su Railway) apre in paper trading. Ogni PollSeconds secondi legge  |
//| /api/ea/pending: per ogni ordine nuovo apre la posizione/pending   |
//| corrispondente con lo stesso entry/SL/TP, poi conferma con         |
//| /api/ea/ack cosi' il server non lo ripropone al giro successivo.   |
//|                                                                    |
//| Non modifica ne' chiude mai posizioni esistenti (proprie o altrui) |
//| — apre solo ordini nuovi. GoldMind continua a fare la propria      |
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
input string SymbolToTrade    = "XAUUSD";     // simbolo oro su questo broker

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
   return ORDER_TIME_SPECIFIED;
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
