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
//+------------------------------------------------------------------+
#property copyright "GoldMind"
#property version   "1.00"

#include <Trade\Trade.mqh>
CTrade trade;

input string ServerUrl     = "https://goldmind-bot-production.up.railway.app"; // URL GoldMind (Railway)
input string ApiToken      = "";           // stesso DASHBOARD_TOKEN configurato su Railway
input int    PollSeconds   = 5;            // ogni quanto controllare nuovi segnali
input double LotSize       = 0.01;         // lotto fisso per ogni copia (demo: non replica il position sizing del bot)
input string SymbolToTrade = "XAUUSD";     // simbolo oro su questo broker

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(990100);
   if(ApiToken == "")
      Print("ATTENZIONE: ApiToken vuoto — /api/ea/pending risponderà 401, nessun ordine verrà copiato.");
   EventSetTimer(PollSeconds);
   Print("GoldMindCopier avviato — polling ", ServerUrl, " ogni ", PollSeconds, "s su ", SymbolToTrade);
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

   if(ot == "BUY")
      sent = trade.Buy(LotSize, SymbolToTrade, 0.0, sl, tp1, cmt);
   else if(ot == "SELL")
      sent = trade.Sell(LotSize, SymbolToTrade, 0.0, sl, tp1, cmt);
   else if(ot == "BUY LIMIT")
      sent = trade.BuyLimit(LotSize, entry, SymbolToTrade, sl, tp1, ORDER_TIME_GTC, 0, cmt);
   else if(ot == "SELL LIMIT")
      sent = trade.SellLimit(LotSize, entry, SymbolToTrade, sl, tp1, ORDER_TIME_GTC, 0, cmt);
   else if(ot == "BUY STOP")
      sent = trade.BuyStop(LotSize, entry, SymbolToTrade, sl, tp1, ORDER_TIME_GTC, 0, cmt);
   else if(ot == "SELL STOP")
      sent = trade.SellStop(LotSize, entry, SymbolToTrade, sl, tp1, ORDER_TIME_GTC, 0, cmt);
   else
   {
      Print("Tipo ordine sconosciuto: '", orderType, "' (trade_id ", tradeId, ") — scartato senza aprire nulla.");
      AckOrder(tradeId);
      return;
   }

   if(sent)
   {
      Print("Copiato: ", ot, " ", SymbolToTrade, " entry=", entry, " sl=", sl, " tp=", tp1, " (", tradeId, ")");
      GlobalVariableSet(seenVar, 1);
   }
   else
   {
      Print("Errore apertura ", ot, " per ", tradeId, ": ", trade.ResultRetcodeDescription());
   }
   AckOrder(tradeId);
}

//+------------------------------------------------------------------+
void AckOrder(string tradeId)
{
   string url  = ServerUrl + "/api/ea/ack?token=" + ApiToken;
   string json = "{\"trade_id\":\"" + tradeId + "\"}";

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
