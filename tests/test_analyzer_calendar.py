"""
Test per il filtro del calendario macro live (analyzer._passes_news_filter,
get_upcoming_events, get_economic_events) — esteso il 2026-09-16 su
richiesta esplicita dell'utente dopo aver perso un movimento reale di
~80 pip su un dato USD arancione (Retail Sales m/m, Medium impact) che il
vecchio filtro (solo rosso+USD) escludeva del tutto.

Verifica sia il nuovo criterio allargato (rosso per USD+major, arancione
solo per USD) sia che get_economic_events() di default resti INVARIATO
(quella funzione alimenta anche econ_risk dentro full_analyze() e il gate
"imminent" di event_driven_strategy — allargarla senza validazione
cambierebbe il comportamento di trading quasi tutti i giorni, vietato da
feedback_conservative_validation_standard).
"""

import os
import sys
import unittest
from datetime import datetime
from unittest import mock

import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import _passes_news_filter, get_upcoming_events, get_economic_events


class TestPassesNewsFilter(unittest.TestCase):
    def test_usd_high_included(self):
        self.assertTrue(_passes_news_filter("high", "usd"))

    def test_usd_medium_included(self):
        """Il caso concreto che ha motivato la richiesta: Retail Sales m/m,
        Medium impact, USD, ~80 pip di movimento reale il 2026-09-16."""
        self.assertTrue(_passes_news_filter("medium", "usd"))

    def test_usd_low_excluded(self):
        self.assertFalse(_passes_news_filter("low", "usd"))

    def test_other_major_currency_high_included(self):
        for currency in ("gbp", "eur", "jpy", "chf", "aud", "cad", "nzd"):
            self.assertTrue(_passes_news_filter("high", currency), currency)

    def test_other_major_currency_medium_excluded(self):
        """Arancione per valute diverse da USD resta escluso — l'oro
        reagisce in modo diretto e affidabile ai dati USD anche di impatto
        medio, per le altre valute un Medium è normalmente troppo
        debole/indiretto e includerle tutte creerebbe rumore."""
        for currency in ("gbp", "eur", "jpy"):
            self.assertFalse(_passes_news_filter("medium", currency), currency)

    def test_minor_currency_excluded_even_if_high(self):
        self.assertFalse(_passes_news_filter("high", "cny"))
        self.assertFalse(_passes_news_filter("high", "try"))

    def test_case_insensitive(self):
        self.assertTrue(_passes_news_filter("HIGH", "USD"))
        self.assertTrue(_passes_news_filter("High", "Gbp"))

    def test_empty_values_excluded(self):
        self.assertFalse(_passes_news_filter("", ""))
        self.assertFalse(_passes_news_filter("high", ""))


def _fake_calendar():
    """Date sempre relative a 'adesso' (ora di New York, come il feed
    FairEconomy reale) — mai hardcoded, altrimenti il test su
    get_economic_events() (che filtra per "oggi") diventerebbe fragile e
    fallirebbe da solo al cambio di giorno."""
    now_et = datetime.now(pytz.timezone("America/New_York"))
    today = now_et.strftime("%Y-%m-%dT")
    return [
        {"date": today + "08:30:00-04:00", "impact": "Medium", "country": "USD",
         "title": "Retail Sales m/m", "forecast": "0.4%", "previous": "0.3%", "actual": ""},
        {"date": today + "08:30:00-04:00", "impact": "Low", "country": "USD",
         "title": "Import Prices m/m", "forecast": "0.1%", "previous": "0.1%", "actual": ""},
        {"date": today + "04:00:00-04:00", "impact": "High", "country": "GBP",
         "title": "CPI y/y", "forecast": "3.0%", "previous": "2.9%", "actual": ""},
        {"date": today + "09:00:00-04:00", "impact": "High", "country": "CNY",
         "title": "GDP y/y", "forecast": "5.0%", "previous": "4.9%", "actual": ""},
    ]


_FAKE_CALENDAR = _fake_calendar()


class TestGetUpcomingEventsBroadFilter(unittest.TestCase):
    """get_upcoming_events() alimenta gli alert Telegram (check_macro_alerts)
    — deve usare SEMPRE il filtro allargato, nessun parametro broad=False
    qui: è esattamente la superficie per cui l'utente ha chiesto la
    copertura completa."""

    def test_medium_usd_and_high_gbp_included_cny_excluded(self):
        with mock.patch("analyzer._fetch_calendar_raw", return_value=_FAKE_CALENDAR):
            events = get_upcoming_events(days_ahead=7, hours_lookback=48)
        titles = {e["title"] for e in events}
        self.assertIn("Retail Sales m/m", titles)
        self.assertIn("CPI y/y", titles)
        self.assertNotIn("Import Prices m/m", titles)
        self.assertNotIn("GDP y/y", titles)

    def test_events_carry_real_impact_and_currency(self):
        with mock.patch("analyzer._fetch_calendar_raw", return_value=_FAKE_CALENDAR):
            events = get_upcoming_events(days_ahead=7, hours_lookback=48)
        by_title = {e["title"]: e for e in events}
        self.assertEqual(by_title["Retail Sales m/m"]["impact"], "MEDIUM")
        self.assertEqual(by_title["Retail Sales m/m"]["currency"], "USD")
        self.assertEqual(by_title["CPI y/y"]["impact"], "HIGH")
        self.assertEqual(by_title["CPI y/y"]["currency"], "GBP")


class TestGetEconomicEventsDefaultUnchanged(unittest.TestCase):
    """get_economic_events() di default (broad=False, il valore usato da
    full_analyze() per econ_risk e da event_driven_strategy) deve
    comportarsi ESATTAMENTE come prima di questa modifica: solo rosso+USD.
    Allargarla senza una validazione statistica dedicata cambierebbe la
    frequenza di econ_risk=True quasi ogni giorno (Medium/USD e High su
    altre valute sono molto più frequenti del solo High/USD)."""

    def test_default_excludes_medium_usd_and_other_currencies(self):
        with mock.patch("analyzer._fetch_calendar_raw", return_value=_FAKE_CALENDAR):
            cal = get_economic_events()
        titles = {e["title"] for e in cal["events"]}
        self.assertNotIn("Retail Sales m/m", titles)
        self.assertNotIn("CPI y/y", titles)
        self.assertNotIn("GDP y/y", titles)

    def test_broad_true_matches_upcoming_events_filter(self):
        with mock.patch("analyzer._fetch_calendar_raw", return_value=_FAKE_CALENDAR):
            cal = get_economic_events(broad=True)
        titles = {e["title"] for e in cal["events"]}
        self.assertIn("CPI y/y", titles)
        self.assertIn("Retail Sales m/m", titles)
        self.assertNotIn("Import Prices m/m", titles)
        self.assertNotIn("GDP y/y", titles)


if __name__ == "__main__":
    unittest.main()
