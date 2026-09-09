"""
Test per il fix del 2026-09-09: blackout dati in produzione con yfinance
rate-limitato, Stooq bloccato da una verifica anti-bot lato loro, e Twelve
Data che esauriva la quota gratuita (800 crediti/giorno) entro
metà mattinata perché ogni fetch falliva su tutte e tre le fonti e
ricascava su Twelve Data ad ogni chiamata.

Due fix distinti coperti qui:
1. get_current_price()/get_dxy_price()/get_us10y_price() avevano una copia
   propria del fallback Twelve Data che NON rispettava
   _twelvedata_blocked_until (impostato da get_data() quando la quota è
   esaurita) — continuavano a chiamare Twelve Data anche a quota già
   esaurita, sprecando altre chiamate. Ora passano tutte per
   _twelvedata_price(), che rispetta lo stesso blocco.
2. _FAIL_BACKOFF alzato da 30s a 300s: con un blackout totale, un backoff
   troppo corto permetteva a più motori/timeframe di ri-cascare su Twelve
   Data più volte nello stesso ciclo da 5 minuti invece di una sola.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer


class _TwelveDataStateResetMixin:
    """Ripristina lo stato globale di analyzer.py prima/dopo ogni test —
    _twelvedata_blocked_until e _price_cache sono moduli-level e persistono
    tra i test se non resettati."""

    def setUp(self):
        self._orig_blocked_until = analyzer._twelvedata_blocked_until
        self._orig_price_cache = dict(analyzer._price_cache)
        analyzer._twelvedata_blocked_until = 0.0
        analyzer._price_cache = {"timestamp": 0, "price": 0.0}

    def tearDown(self):
        analyzer._twelvedata_blocked_until = self._orig_blocked_until
        analyzer._price_cache = self._orig_price_cache


class TestTwelveDataBlockHelpers(_TwelveDataStateResetMixin, unittest.TestCase):
    def test_available_by_default(self):
        self.assertTrue(analyzer._twelvedata_available())

    def test_mark_blocked_makes_unavailable_until_utc_midnight(self):
        analyzer._mark_twelvedata_blocked()
        self.assertFalse(analyzer._twelvedata_available())
        # Il blocco deve durare fino a mezzanotte UTC, non un breve backoff fisso.
        remaining_h = (analyzer._twelvedata_blocked_until - time.time()) / 3600
        self.assertGreater(remaining_h, 0)
        self.assertLessEqual(remaining_h, 24)

    def test_quota_exceeded_detection_accepts_string_or_exception(self):
        msg = "You have run out of API credits for the day. 884 used, limit 800."
        self.assertTrue(analyzer._twelvedata_quota_exceeded(msg))
        self.assertTrue(analyzer._twelvedata_quota_exceeded(ValueError(msg)))
        self.assertFalse(analyzer._twelvedata_quota_exceeded("nessun dato per 1h"))


class TestTwelveDataPriceHelper(_TwelveDataStateResetMixin, unittest.TestCase):
    def test_skips_request_entirely_when_blocked(self):
        analyzer._mark_twelvedata_blocked()
        with patch("analyzer.requests.get") as mock_get:
            price = analyzer._twelvedata_price("XAU/USD")
        mock_get.assert_not_called()
        self.assertEqual(price, 0.0)

    def test_marks_blocked_on_quota_exceeded_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": 429,
            "message": "You have run out of API credits for the day.",
        }
        with patch("analyzer.requests.get", return_value=mock_resp):
            price = analyzer._twelvedata_price("XAU/USD")
        self.assertEqual(price, 0.0)
        self.assertFalse(analyzer._twelvedata_available())

    def test_returns_price_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"price": "4400.5"}
        with patch("analyzer.requests.get", return_value=mock_resp):
            price = analyzer._twelvedata_price("XAU/USD")
        self.assertEqual(price, 4400.5)
        self.assertTrue(analyzer._twelvedata_available())


class TestGetCurrentPriceRespectsBlock(_TwelveDataStateResetMixin, unittest.TestCase):
    def test_does_not_call_twelvedata_when_yfinance_fails_and_quota_blocked(self):
        analyzer._mark_twelvedata_blocked()
        with patch("yfinance.Ticker", side_effect=Exception("rate limited")), \
             patch("analyzer.requests.get") as mock_get:
            price = analyzer.get_current_price()
        mock_get.assert_not_called()
        self.assertEqual(price, 0.0)

    def test_falls_back_to_twelvedata_when_not_blocked(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"price": "4401.2"}
        with patch("yfinance.Ticker", side_effect=Exception("rate limited")), \
             patch("analyzer.requests.get", return_value=mock_resp):
            price = analyzer.get_current_price()
        self.assertEqual(price, 4401.2)


class TestGetDataSkipsTwelveDataWhenBlocked(_TwelveDataStateResetMixin, unittest.TestCase):
    def test_twelvedata_excluded_from_sources_when_blocked(self):
        analyzer._mark_twelvedata_blocked()
        analyzer._data_cache.clear()
        analyzer._data_fail_cache.clear()
        with patch("analyzer._fetch_yfinance", side_effect=ValueError("yfinance down")), \
             patch("analyzer._fetch_stooq", side_effect=ValueError("stooq down")), \
             patch("analyzer._fetch_twelvedata") as mock_td, \
             patch("analyzer.time.sleep"):
            with self.assertRaises(ValueError):
                analyzer.get_data(interval="1h", outputsize=50, bypass_cache=True)
        mock_td.assert_not_called()


class TestFailBackoffDuration(unittest.TestCase):
    def test_fail_backoff_matches_auto_check_cycle_not_old_30s(self):
        """Regressione: _FAIL_BACKOFF era 30s, troppo corto per evitare che
        più motori/timeframe nello stesso ciclo da 5 minuti ricascassero
        ripetutamente su Twelve Data durante un blackout totale."""
        self.assertGreaterEqual(analyzer._FAIL_BACKOFF, 240)


if __name__ == "__main__":
    unittest.main()
