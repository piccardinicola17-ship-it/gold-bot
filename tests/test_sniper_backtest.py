"""
Test per sniper_backtest.py — harness permanente introdotto il 2026-09-15
dopo che i backtest cecchino del 2026-09-14 (script ad-hoc in scratchpad)
non si sono più potuti riprodurre. Copre in particolare i due bug reali
trovati quel giorno con la versione scratchpad:
  1. segnale non edge-triggered -> un trade chiuso rapidamente riapriva
     subito sulla stessa struttura SMC ancora persistente (n gonfiato)
  2. _stat_arb_for_date con un Timestamp tz-aware ritornava sempre NEUTRAL
     (confronto silenzioso fallito contro uno storico DXY/TLT tz-naive)
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sniper_backtest as sb


def _flat_ohlc(n: int, base: float = 2000.0, start: str = "2024-01-01") -> pd.DataFrame:
    """n barre giornaliere piatte (nessun movimento) - sufficienti a far
    girare compute_indicators senza NaN dopo il warmup, senza generare
    trigger di SL/TP finché un test non li introduce esplicitamente."""
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {"Open": base, "High": base + 0.5, "Low": base - 0.5, "Close": base, "Volume": 100.0},
        index=idx,
    )


class TestStatsFor(unittest.TestCase):
    def test_pf_and_win_rate(self):
        trades = [
            {"r_result": 2.0, "outcome": "WIN_TP2"},
            {"r_result": -1.0, "outcome": "LOSS"},
            {"r_result": -1.0, "outcome": "LOSS"},
            {"r_result": 3.0, "outcome": "WIN_TP3"},
        ]
        stats = sb.stats_for(trades)
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["pf"], 2.5)  # (2+3) / (1+1)
        self.assertEqual(stats["wr"], 50.0)
        self.assertEqual(stats["total_r"], 3.0)

    def test_no_losses_gives_infinite_pf(self):
        trades = [{"r_result": 1.0, "outcome": "WIN_TP1"}]
        self.assertEqual(sb.stats_for(trades)["pf"], float("inf"))

    def test_empty_trades(self):
        stats = sb.stats_for([])
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["wr"], 0.0)


class TestSplitStats(unittest.TestCase):
    def test_splits_chronologically_not_by_insertion_order(self):
        trades = [
            {"r_result": -1.0, "outcome": "LOSS", "time": pd.Timestamp("2024-03-01")},
            {"r_result": 2.0, "outcome": "WIN_TP2", "time": pd.Timestamp("2024-01-01")},
            {"r_result": 2.0, "outcome": "WIN_TP2", "time": pd.Timestamp("2024-02-01")},
            {"r_result": -1.0, "outcome": "LOSS", "time": pd.Timestamp("2024-04-01")},
        ]
        half1, half2 = sb.split_stats(trades)
        # Meta' 1 cronologica = gen+feb (2 win), meta' 2 = mar+apr (2 loss)
        self.assertEqual(half1["wr"], 100.0)
        self.assertEqual(half2["wr"], 0.0)


class TestValidated(unittest.TestCase):
    def _trades(self, n_wins, n_losses, time_start="2024-01-01"):
        times = pd.date_range(time_start, periods=n_wins + n_losses, freq="D")
        trades = [{"r_result": 2.0, "outcome": "WIN_TP2", "time": t} for t in times[:n_wins]]
        trades += [{"r_result": -1.0, "outcome": "LOSS", "time": t} for t in times[n_wins:]]
        return trades

    def test_validated_when_pf_over_1_on_both_halves(self):
        trades = self._trades(30, 10)  # PF alto e stabile su entrambe le meta'
        result = sb.validated(trades, min_n=10)
        self.assertEqual(result["verdict"], "VALIDATO")
        self.assertTrue(result["split_holds"])

    def test_not_validated_when_a_half_fails(self):
        wins_first_half = [
            {"r_result": 2.0, "outcome": "WIN_TP2", "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)}
            for i in range(15)
        ]
        losses_second_half = [
            {"r_result": -1.0, "outcome": "LOSS", "time": pd.Timestamp("2024-02-01") + pd.Timedelta(days=i)}
            for i in range(15)
        ]
        trades = wins_first_half + losses_second_half
        result = sb.validated(trades, min_n=10)
        self.assertEqual(result["verdict"], "NON VALIDATO")

    def test_insufficient_sample_per_half_is_not_a_pass(self):
        trades = self._trades(3, 1)  # troppo pochi per meta' (min_n=20 default)
        result = sb.validated(trades)
        self.assertEqual(result["verdict"], "campione insufficiente per uno split affidabile")


class TestSecondSignal(unittest.TestCase):
    def test_stat_arb_strips_timezone_before_lookup(self):
        """Bug reale del 2026-09-14: senza lo strip, _stat_arb_for_date
        riceveva un Timestamp tz-aware e ritornava sempre NEUTRAL."""
        tz_aware_ts = pd.Timestamp("2024-06-15 13:30:00+00:00")
        window = _flat_ohlc(5)
        with mock.patch("sniper_backtest._stat_arb_for_date") as mock_fn:
            mock_fn.return_value = {"signal": "BUY", "score": 4}
            sb._second_signal("stat_arb", window, tz_aware_ts)
        called_with = mock_fn.call_args[0][0]
        self.assertIsNone(called_with.tzinfo, "il Timestamp passato a _stat_arb_for_date deve essere tz-naive")

    def test_stat_arb_passes_naive_timestamp_unchanged(self):
        naive_ts = pd.Timestamp("2024-06-15 13:30:00")
        window = _flat_ohlc(5)
        with mock.patch("sniper_backtest._stat_arb_for_date") as mock_fn:
            mock_fn.return_value = {"signal": "NEUTRAL", "score": 0}
            sb._second_signal("stat_arb", window, naive_ts)
        mock_fn.assert_called_once_with(naive_ts)

    def test_candlestick_delegates_to_analyzer(self):
        window = _flat_ohlc(5)
        with mock.patch("sniper_backtest.candlestick_strategy") as mock_fn:
            mock_fn.return_value = {"signal": "SELL", "score": 2}
            result = sb._second_signal("candlestick", window, pd.Timestamp("2024-01-01"))
        self.assertEqual(result["signal"], "SELL")

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            sb._second_signal("bogus", _flat_ohlc(5), pd.Timestamp("2024-01-01"))


class TestRunSniperBacktestEdgeTrigger(unittest.TestCase):
    """Il bug centrale del 2026-09-14: senza edge-trigger, un segnale SMC
    che resta BUY per molti bar consecutivi apriva un nuovo trade ad ogni
    bar libero, non solo quando la direzione CAMBIA."""

    def setUp(self):
        self.df = _flat_ohlc(300)

    def test_persisting_same_direction_signal_opens_only_one_trade(self):
        # SMC sempre BUY, MAI cambia direzione, dal bar 220 in poi. SL
        # vicinissimo: ogni trade chiude in LOSS al bar successivo, quindi
        # senza edge-trigger si riaprirebbe subito sulla stessa struttura
        # ancora "BUY" — decine di volte sui restanti ~80 bar. Con
        # l'edge-trigger, ne deve aprire esattamente UNO: dopo la prima
        # chiusura il segnale e' ancora "BUY" (invariato), quindi non e'
        # un nuovo edge.
        with mock.patch("sniper_backtest.smc_generic_zone_signal", return_value={"signal": "BUY", "score": 8}), \
             mock.patch("sniper_backtest.calculate_risk_levels") as mock_risk, \
             mock.patch("sniper_backtest.detect_market_regime", return_value={"regime": "NORMAL"}):
            mock_risk.return_value = {"sl": 1999.6, "tp1": 2100.0, "tp2": 2200.0, "tp3": 2300.0}
            trades = sb.run_sniper_backtest(self.df, "1day", "pure")
        self.assertEqual(len(trades), 1)

    def test_signal_flip_after_close_opens_a_new_trade(self):
        """Se il segnale CAMBIA direzione dopo che il primo trade e'
        chiuso, un nuovo trade deve aprirsi - l'edge-trigger blocca solo la
        RIPETIZIONE dello stesso segnale, non un segnale genuinamente
        nuovo."""
        call_count = {"n": 0}

        def alternating_signal(window):
            call_count["n"] += 1
            # BUY per le prime chiamate (fino alla chiusura per SL), poi SELL
            return {"signal": "BUY" if call_count["n"] < 3 else "SELL", "score": 8}

        with mock.patch("sniper_backtest.smc_generic_zone_signal", side_effect=alternating_signal), \
             mock.patch("sniper_backtest.calculate_risk_levels") as mock_risk, \
             mock.patch("sniper_backtest.detect_market_regime", return_value={"regime": "NORMAL"}):
            # SL vicinissimo: il trade BUY chiude in LOSS al bar successivo
            # (Low del flat OHLC scende sempre di 0.5 sotto Close).
            mock_risk.return_value = {"sl": 1999.6, "tp1": 2100.0, "tp2": 2200.0, "tp3": 2300.0}
            trades = sb.run_sniper_backtest(self.df, "1day", "pure")
        signals = [t["signal"] for t in trades]
        self.assertIn("BUY", signals)
        self.assertIn("SELL", signals)


class TestRunSniperBacktestSessionStop(unittest.TestCase):
    def test_stops_after_max_consecutive_losses_same_day(self):
        # Segnale che alterna BUY/SELL ad ogni chiamata: ogni volta e' un
        # edge NUOVO (mai la stessa direzione due volte di fila), quindi
        # l'edge-trigger non lo sopprime — isola cosi' lo stop-sessione da
        # testare separatamente. SL fisso vicinissimo chiude ogni trade in
        # LOSS al bar successivo (sia BUY che SELL, stesso OHLC piatto).
        # 250 barre orarie nello stesso giorno solare: dopo
        # MAX_CONSECUTIVE_LOSS perdite consecutive lo stesso giorno, la
        # sessione si ferma e non apre piu' trade fino al giorno dopo.
        idx = pd.date_range("2024-06-01 00:00", periods=250, freq="h")
        df = pd.DataFrame(
            {"Open": 2000.0, "High": 2000.5, "Low": 1999.5, "Close": 2000.0, "Volume": 100.0}, index=idx
        )
        toggle = {"buy": True}

        def alternating_signal(window):
            toggle["buy"] = not toggle["buy"]
            return {"signal": "BUY" if toggle["buy"] else "SELL", "score": 8}

        with mock.patch("sniper_backtest.smc_generic_zone_signal", side_effect=alternating_signal), \
             mock.patch("sniper_backtest.calculate_risk_levels") as mock_risk, \
             mock.patch("sniper_backtest.detect_market_regime", return_value={"regime": "NORMAL"}):
            mock_risk.return_value = {"sl": 1999.6, "tp1": 2100.0, "tp2": 2200.0, "tp3": 2300.0}
            trades = sb.run_sniper_backtest(df, "1h", "pure")
        first_day = pd.Timestamp("2024-06-01").date()
        trades_first_day = [t for t in trades if t["time"].date() == first_day]
        from risk_manager import MAX_CONSECUTIVE_LOSS
        self.assertTrue(all(t["outcome"] == "LOSS" for t in trades_first_day))
        self.assertLessEqual(len(trades_first_day), MAX_CONSECUTIVE_LOSS)
        # E deve aver ripreso il giorno dopo (altrimenti il test sarebbe
        # falsato da un altro motivo, es. un bug che blocca tutto per sempre)
        self.assertGreater(len(trades), len(trades_first_day))


if __name__ == "__main__":
    unittest.main()
