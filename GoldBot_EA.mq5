//+------------------------------------------------------------------+
//|  GoldBot_EA.mq5                                                  |
//|  Legge il file signal.json e apre ordini su XAU/USD              |
//+------------------------------------------------------------------+
#property copyright "Gold Trading Bot"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

input string   SignalFilePath  = "C:\\Users\\nico.piccardi\\Documents\\MT5_Signal\\signal.json";
input double   LotSize         = 0.01;
input double   MaxSlippagePips = 10.0;
input int      MagicNumber     = 20260101;

CTrade trade;
string lastExecutedTime = "";

int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   Print("GoldBot EA v2 avviato. File: ", SignalFilePath);
   EventSetTimer(2); // Controlla ogni 2 secondi invece di aspettare i tick
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   CheckAndExecuteSignal();
}

void OnTick()
{
   CheckAndExecuteSignal();
}

void CheckAndExecuteSignal()
{
   int fileHandle = FileOpen(SignalFilePath, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fileHandle == INVALID_HANDLE)
   {
      Print("File non trovato: ", SignalFilePath, " Errore: ", GetLastError());
      return;
   }

   string content = "";
   while(!FileIsEnding(fileHandle))
      content += FileReadString(fileHandle);
   FileClose(fileHandle);

   if(content == "") { Print("File vuoto"); return; }

   string signal   = ExtractJsonString(content, "signal");
   string timeStr  = ExtractJsonString(content, "time");
   double price    = ExtractJsonDouble(content, "price");
   double tp       = ExtractJsonDouble(content, "tp");
   double sl       = ExtractJsonDouble(content, "sl");
   bool   executed = ExtractJsonBool(content, "executed");

   Print("Letto: signal=", signal, " time=", timeStr, " executed=", executed);

   if(executed || timeStr == lastExecutedTime || timeStr == "")
   {
      Print("Segnale gia eseguito o vuoto, skip");
      return;
   }

   double currentPrice = 0;
   if(signal == "BUY")
      currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   else if(signal == "SELL")
      currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   else { Print("Segnale non valido: ", signal); return; }

   double pipSize   = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 10;
   double priceDiff = MathAbs(currentPrice - price) / pipSize;

   Print("Prezzo segnale: ", price, " Prezzo attuale: ", currentPrice, " Diff pips: ", priceDiff);

   if(priceDiff > MaxSlippagePips)
   {
      Print("Segnale scartato: prezzo mosso di ", DoubleToString(priceDiff, 1), " pips (max ", MaxSlippagePips, ")");
      MarkAsExecuted(content);
      lastExecutedTime = timeStr;
      return;
   }

   bool result = false;
   if(signal == "BUY")
      result = trade.Buy(LotSize, _Symbol, 0, sl, tp, "GoldBot BUY");
   else if(signal == "SELL")
      result = trade.Sell(LotSize, _Symbol, 0, sl, tp, "GoldBot SELL");

   if(result)
   {
      Print("Ordine aperto: ", signal, " @ ", currentPrice, " TP:", tp, " SL:", sl);
      lastExecutedTime = timeStr;
      MarkAsExecuted(content);
   }
   else
      Print("Errore apertura ordine: ", GetLastError(), " RetCode: ", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
}

void MarkAsExecuted(string content)
{
   StringReplace(content, "\"executed\": false", "\"executed\": true");
   StringReplace(content, "\"executed\":false", "\"executed\":true");
   int fileHandle = FileOpen(SignalFilePath, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fileHandle != INVALID_HANDLE)
   {
      FileWriteString(fileHandle, content);
      FileClose(fileHandle);
   }
}

string ExtractJsonString(string json, string key)
{
   string search = "\"" + key + "\": \"";
   int start = StringFind(json, search);
   if(start == -1) { search = "\"" + key + "\":\""; start = StringFind(json, search); }
   if(start == -1) return "";
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end == -1) return "";
   return StringSubstr(json, start, end - start);
}

double ExtractJsonDouble(string json, string key)
{
   string search = "\"" + key + "\": ";
   int start = StringFind(json, search);
   if(start == -1) { search = "\"" + key + "\":"; start = StringFind(json, search); }
   if(start == -1) return 0;
   start += StringLen(search);
   int end = StringFind(json, ",", start);
   if(end == -1) end = StringFind(json, "}", start);
   if(end == -1) return 0;
   string val = StringSubstr(json, start, end - start);
   StringTrimLeft(val);
   StringTrimRight(val);
   return StringToDouble(val);
}

bool ExtractJsonBool(string json, string key)
{
   string search = "\"" + key + "\": ";
   int start = StringFind(json, search);
   if(start == -1) { search = "\"" + key + "\":"; start = StringFind(json, search); }
   if(start == -1) return false;
   start += StringLen(search);
   string val = StringSubstr(json, start, 5);
   StringTrimLeft(val);
   return StringFind(val, "true") == 0;
}
//+------------------------------------------------------------------+
