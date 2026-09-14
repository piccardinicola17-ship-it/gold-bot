"""
groq_client.py — meccanica HTTP condivisa per le chiamate a Groq (chat
completions), usata da news_analyst.py, self_learning.py e ai_assistant.py.

Prima ogni file aveva la propria copia identica di endpoint/modello/
costruzione della richiesta/parsing della risposta — già leggermente
divergenti tra loro (timeout 25/25/20, temperatura 0.3/0.2/0.4), col rischio
che un domani (cambio modello Groq, nuovo parametro richiesto) qualcuno
aggiornasse una copia e dimenticasse le altre due.

Il controllo "manca la chiave" e il testo di fallback restano nel file
chiamante (non qui): ognuno ha il proprio fallback e i test patchano
GROQ_API_KEY sul singolo modulo (es. ai_assistant.GROQ_API_KEY) — spostare
anche quel controllo qui romperebbe quei patch, che agiscono sull'attributo
del modulo chiamante, non su questo. Qui vive solo la parte davvero
identica: come si parla con Groq.
"""
import logging

import requests

logger = logging.getLogger(__name__)

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "groq/compound-mini"


def chat(api_key: str, messages: list, *, max_tokens: int = 500,
         temperature: float = 0.3, timeout: int = 25) -> str:
    """Chiamata Groq chat completions. Propaga qualunque eccezione di rete/
    HTTP/parsing — il chiamante decide il proprio messaggio di fallback."""
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
