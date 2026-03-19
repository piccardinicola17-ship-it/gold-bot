# 🥇 Gold Trading Bot — Guida Setup

## 1. Installa le dipendenze
```bash
cd gold_bot
pip install -r requirements.txt
```

## 2. Configura il bot
Apri `gold_bot.py` e modifica le prime righe:

```python
TELEGRAM_TOKEN = "IL_TUO_TOKEN_QUI"   # Token da @BotFather
CHAT_ID        = "IL_TUO_CHAT_ID_QUI" # Vedi passo 3
CHECK_INTERVAL = 15                    # Ogni quanti minuti controlla
```

## 3. Ottieni il tuo Chat ID
1. Avvia il bot su Telegram
2. Scrivi /start
3. Il bot ti risponde con il tuo Chat ID
4. Copia quel numero e incollalo in CHAT_ID

## 4. Avvia il bot
```bash
python gold_bot.py
```

## Comandi disponibili
| Comando | Funzione |
|---------|----------|
| /start | Mostra Chat ID e guida |
| /signal | Analisi manuale immediata |
| /status | Parametri e stato del bot |

## Come funziona la logica
- **BUY** → EMA20 > EMA50 + RSI < 65 + MACD positivo
- **SELL** → EMA20 < EMA50 + RSI > 35 + MACD negativo
- **TP** = prezzo ± (ATR × 3.0)
- **SL** = prezzo ± (ATR × 1.5)
- **Risk/Reward** = 1:2

## Tenerlo sempre attivo (opzionale)
Usa **Railway.app** (gratuito):
1. Crea account su railway.app
2. Carica la cartella gold_bot
3. Imposta variabili d'ambiente: TELEGRAM_TOKEN e CHAT_ID
4. Deploy → il bot gira 24/7

⚠️ Questo bot è a scopo educativo. Non è consulenza finanziaria.
