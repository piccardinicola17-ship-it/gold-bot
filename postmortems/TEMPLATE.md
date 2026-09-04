# Post-mortem: <titolo breve>

**Data**: AAAA-MM-GG
**Trade/i coinvolti**: trade_id (se applicabile)
**Commit del fix**: hash breve

## Cosa è successo

Descrizione oggettiva dei fatti: cosa ha fatto il bot, cosa ci si aspettava, quale trade/decisione è risultato sbagliato.

## Causa radice

Non il sintomo — la causa. Se sono coinvolti due meccanismi che dovevano restare sincronizzati e non lo erano, dirlo esplicitamente (è il pattern più comune in questo codebase).

## Come è stato trovato

Osservazione diretta in produzione? Audit proattivo? Segnalazione dell'utente?

## Fix applicato

Cosa è cambiato, in quale file, perché quella e non un'altra soluzione.

## Come prevenire la ricorrenza

- Test aggiunto? Quale scenario copre.
- C'è un'altra implementazione della stessa logica altrove che rischia la stessa deriva? (grep per pattern simili prima di chiudere)

## Impatto

Quantificato se possibile: quanti R persi, quanti trade coinvolti, per quanto tempo il bug è stato attivo prima di essere trovato.
