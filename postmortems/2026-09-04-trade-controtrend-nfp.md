# Post-mortem: trade H4 controtrend preso in pieno dall'NFP

**Data**: 2026-09-04
**Trade coinvolto**: id 83 / trade_id `506d376a-1746-44e0-82c4-b7782f3f6897` (H4, BUY)
**Commit del fix**: `aa53e6e` (chiusura protettiva automatica) + `cae5474` (endpoint di correzione retroattiva)

## Cosa è successo

Un trade H4 BUY, aperto il giorno prima dell'NFP, era ancora attivo al momento del rilascio. Il bot aveva già calcolato un bias SELL per l'evento (poi confermato dal movimento di prezzo reale) tramite `analyze_macro_event()`, ma non esisteva alcun meccanismo che confrontasse quel bias con i trade già aperti. Il prezzo si è mosso nella direzione SELL prevista e il trade H4, controtrend, ha preso lo stop loss pochi minuti dopo il rilascio.

## Causa radice

Non un bug di calcolo: una funzionalità mancante. Il bot generava un bias pre-evento corretto ma lo usava solo per il messaggio di alert, senza mai agire su di esso rispetto ai trade già in corso. Nessun collegamento esisteva tra "il bot sa che sta per succedere qualcosa in direzione X" e "il bot ha già un trade aperto in direzione opposta a X".

## Come è stato trovato

Osservazione diretta in produzione durante l'NFP del 2026-09-04, segnalata dall'utente controllando la dashboard subito dopo l'evento.

## Fix applicato

`aa53e6e`: nel blocco pre-evento di `check_macro_alerts()`, se il bias è direzionale (BUY/SELL, non NEUTRO), ogni trade aperto in direzione opposta viene ora chiuso automaticamente (se già attivo, con R reale calcolato proporzionalmente su entry/sl/exit) o cancellato (se ancora pending, nessun capitale virtuale a rischio). Il trade già perso quel giorno non poteva beneficiarne retroattivamente: `cae5474` ha aggiunto un endpoint amministrativo (`/api/correct-trade`) per correggerlo a posteriori — chiuso a 4469 (il prezzo poco prima dell'evento) come se la protezione fosse già esistita, con i pip ricalcolati rispetto all'entry reale.

## Come prevenire la ricorrenza

- 5 test aggiunti (`TestProtectiveCloseAgainstEventBias`) coprono: trade attivo controtrend chiuso, pending controtrend cancellato, trade nella stessa direzione del bias lasciato intatto, bias NEUTRO che non tocca nulla.
- Non risultava esistere un secondo punto del codice che duplicasse questa logica (verificato con grep su "bias" e "pre_event" in `gold_bot.py`) — il rischio di deriva futura è quindi basso finché resta un solo punto di scrittura.

## Impatto

Un trade, esito LOSS reale (-1R) diventato CLOSED_EARLY dopo la correzione (R proporzionale, molto meno negativo). Il bug era strutturalmente presente dal giorno in cui `analyze_macro_event()` è stato introdotto, ma è emerso solo con il primo evento macro ad alto impatto osservato dopo quel deploy.
