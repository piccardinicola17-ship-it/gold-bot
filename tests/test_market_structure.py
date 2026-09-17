"""
Test per market_structure.py — liquidità (swing high/low), FVG/inefficienze
e order block calcolati in modo deterministico su OHLC sintetici (nessuna
dipendenza da dati di mercato reali, per rendere i test riproducibili).

Aggiunto il 2026-09-17 insieme al modulo, su richiesta esplicita
dell'utente di arricchire i messaggi macro con un'analisi tecnica a
grafico "vera" (numeri calcolati, non un racconto dell'AI).
"""

import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market_structure as ms


def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """rows: lista di (open, high, low, close) — l'indice è un range di
    minuti crescente, il valore esatto non conta per questi test."""
    index = pd.date_range("2026-09-01", periods=len(rows), freq="15min")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)
    return df.astype(float)


class TestSwingHighsLows(unittest.TestCase):
    def test_finds_a_clear_swing_high(self):
        rows = [(100, 101, 99, 100)] * 3 + [(100, 110, 100, 105)] + [(100, 101, 99, 100)] * 3
        df = _make_df(rows)
        swings = ms._find_swing_highs(df, window=3)
        self.assertEqual(len(swings), 1)
        self.assertEqual(swings[0][1], 110)

    def test_finds_a_clear_swing_low(self):
        rows = [(100, 101, 99, 100)] * 3 + [(95, 96, 90, 95)] + [(100, 101, 99, 100)] * 3
        df = _make_df(rows)
        swings = ms._find_swing_lows(df, window=3)
        self.assertEqual(len(swings), 1)
        self.assertEqual(swings[0][1], 90)

    def test_excludes_unconfirmed_recent_bars(self):
        """Un massimo nelle ultime `window` candele non è ancora
        confermato (potrebbe essere superato subito dopo) — non deve
        comparire tra gli swing."""
        rows = [(100, 101, 99, 100)] * 5 + [(100, 120, 100, 110)]
        df = _make_df(rows)
        swings = ms._find_swing_highs(df, window=3)
        self.assertEqual(swings, [])


class TestNearestLiquidity(unittest.TestCase):
    def test_reports_closest_swing_above_and_below(self):
        rows = (
            [(100, 101, 99, 100)] * 3
            + [(100, 108, 100, 104)]          # swing high a 108
            + [(100, 101, 99, 100)] * 3
            + [(100, 101, 92, 98)]            # swing low a 92
            + [(100, 101, 99, 100)] * 3
        )
        df = _make_df(rows)
        liq = ms._nearest_liquidity(df, current_price=100.0, window=3)
        self.assertEqual(liq["above"], 108)
        self.assertEqual(liq["below"], 92)

    def test_no_liquidity_found_returns_none(self):
        rows = [(100, 101, 99, 100)] * 10
        df = _make_df(rows)
        liq = ms._nearest_liquidity(df, current_price=100.0, window=3)
        self.assertIsNone(liq["above"])
        self.assertIsNone(liq["below"])


class TestFairValueGap(unittest.TestCase):
    def test_detects_bullish_fvg(self):
        # candela 0: high=100 — candela 1: impulso — candela 2: low=105 (>100 -> gap)
        rows = [(99, 100, 98, 100), (100, 106, 100, 105), (105, 107, 105, 106)]
        df = _make_df(rows)
        fvgs = ms._find_fvgs(df)
        self.assertEqual(len(fvgs), 1)
        self.assertEqual(fvgs[0]["type"], "bullish")
        self.assertEqual(fvgs[0]["zone_low"], 100)
        self.assertEqual(fvgs[0]["zone_high"], 105)

    def test_detects_bearish_fvg(self):
        rows = [(101, 102, 100, 101), (100, 100, 94, 95), (94, 95, 93, 94)]
        df = _make_df(rows)
        fvgs = ms._find_fvgs(df)
        self.assertEqual(len(fvgs), 1)
        self.assertEqual(fvgs[0]["type"], "bearish")
        self.assertEqual(fvgs[0]["zone_low"], 95)
        self.assertEqual(fvgs[0]["zone_high"], 100)

    def test_no_gap_no_fvg(self):
        rows = [(100, 102, 98, 100), (100, 103, 97, 100), (100, 102, 98, 100)]
        df = _make_df(rows)
        self.assertEqual(ms._find_fvgs(df), [])

    def test_fvg_stays_open_if_never_revisited(self):
        rows = [
            (99, 100, 98, 100), (100, 106, 100, 105), (105, 107, 105, 106),
            (106, 109, 106, 108), (108, 110, 108, 109),
        ]
        df = _make_df(rows)
        fvg = ms._find_fvgs(df)[0]
        self.assertTrue(ms._fvg_is_open(df, fvg))

    def test_fvg_closes_once_price_trades_back_into_zone(self):
        rows = [
            (99, 100, 98, 100), (100, 106, 100, 105), (105, 107, 105, 106),
            (106, 106, 101, 102),  # ritraccia dentro la zona 100-105
        ]
        df = _make_df(rows)
        fvg = ms._find_fvgs(df)[0]
        self.assertFalse(ms._fvg_is_open(df, fvg))

    def test_last_open_fvg_picks_most_recent(self):
        rows = [
            (99, 100, 98, 100),      # idx0
            (100, 106, 100, 105),    # idx1 -> fvg #1 (idx0,1,2): zona 100-105, poi richiusa
            (105, 107, 105, 106),    # idx2
            (106, 103, 101, 102),    # idx3
            (102, 102, 100, 101),    # idx4
            (101, 102, 100, 101),    # idx5
            (102, 116, 102, 115),    # idx6 -> fvg #2 (idx5,6,7): zona 102-115, più recente e mai richiusa
            (115, 117, 115, 116),    # idx7
        ]
        df = _make_df(rows)
        last_fvg = ms._last_open_fvg(df, "bullish")
        self.assertIsNotNone(last_fvg)
        self.assertEqual(last_fvg["idx"], 6)
        self.assertEqual(last_fvg["zone_low"], 102)
        self.assertEqual(last_fvg["zone_high"], 115)


class TestOrderBlock(unittest.TestCase):
    def test_detects_bullish_order_block_before_strong_up_move(self):
        # 14 barre piatte per costruire l'ATR, poi una candela ribassista
        # seguita da un impulso rialzista molto più ampio del range medio.
        flat = [(100, 100.5, 99.5, 100)] * 14
        down_candle = (100, 100.2, 98.5, 99)      # candela "opposta"
        impulse = (99, 115, 99, 114)               # impulso forte rialzista
        rows = flat + [down_candle, impulse]
        df = _make_df(rows)
        ob = ms._find_last_order_block(df, "bullish")
        self.assertIsNotNone(ob)
        self.assertEqual(ob["low"], 99)
        self.assertEqual(ob["high"], 100)

    def test_no_strong_move_no_order_block(self):
        rows = [(100, 100.5, 99.5, 100)] * 20
        df = _make_df(rows)
        self.assertIsNone(ms._find_last_order_block(df, "bullish"))
        self.assertIsNone(ms._find_last_order_block(df, "bearish"))

    def test_too_short_dataframe_returns_none(self):
        df = _make_df([(100, 101, 99, 100)] * 5)
        self.assertIsNone(ms._find_last_order_block(df, "bullish"))


class TestFormatMarketStructure(unittest.TestCase):
    def test_includes_only_available_sections(self):
        structure = {
            "liquidity": {"above": 4350.0, "below": None},
            "fvg_bullish": None,
            "fvg_bearish": {"zone_low": 4300.0, "zone_high": 4305.0},
            "order_block_bullish": None,
            "order_block_bearish": None,
        }
        text = ms.format_market_structure(structure, current_price=4320.0)
        self.assertIn("Liquidità sopra: $4350.00", text)
        self.assertNotIn("Liquidità sotto", text)
        self.assertIn("FVG ribassista aperta: $4300.00", text)
        self.assertNotIn("FVG rialzista", text)
        self.assertNotIn("Order block", text)

    def test_empty_structure_returns_empty_string(self):
        structure = {
            "liquidity": {"above": None, "below": None},
            "fvg_bullish": None, "fvg_bearish": None,
            "order_block_bullish": None, "order_block_bearish": None,
        }
        self.assertEqual(ms.format_market_structure(structure, current_price=4320.0), "")


class TestGetMarketStructureSnapshot(unittest.TestCase):
    def test_returns_empty_string_on_data_failure(self):
        with mock.patch("analyzer.get_data", side_effect=ValueError("no data")):
            self.assertEqual(ms.get_market_structure_snapshot(4320.0), "")

    def test_returns_empty_string_when_too_few_candles(self):
        with mock.patch("analyzer.get_data", return_value=_make_df([(100, 101, 99, 100)] * 5)):
            self.assertEqual(ms.get_market_structure_snapshot(4320.0), "")

    def test_returns_formatted_text_on_success(self):
        rows = (
            [(100, 101, 99, 100)] * 3
            + [(100, 108, 100, 104)]
            + [(100, 101, 99, 100)] * 30
        )
        with mock.patch("analyzer.get_data", return_value=_make_df(rows)):
            text = ms.get_market_structure_snapshot(current_price=100.0)
        self.assertIn("Struttura di mercato", text)
        self.assertIn("Liquidità sopra", text)


if __name__ == "__main__":
    unittest.main()
