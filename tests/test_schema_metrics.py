import math

from typed_decisions.schema import readout, entropy_confidence
from typed_decisions.metrics import summarize, ece, fit_temperature


def test_readout_kinds():
    p = [0.1, 0.7, 0.2]
    assert readout("choice", p)["choice"] == 1
    assert abs(readout("score", p)["score"] - 1.1) < 1e-9  # 0*0.1 + 1*0.7 + 2*0.2
    assert readout("noul", [0.3, 0.7])["noul"] == 0.7
    assert entropy_confidence([1.0, 0.0]) == 1.0
    assert abs(entropy_confidence([0.5, 0.5])) < 1e-9


def test_ece_and_empty_groups():
    assert ece([], []) is None
    # perfectly calibrated: conf 0.8 right 80% of the time
    confs = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert abs(ece(confs, correct)) < 1e-9
    # miscalibrated: conf 0.9, always wrong -> ECE 0.9
    assert abs(ece([0.9] * 5, [False] * 5) - 0.9) < 1e-9
    assert summarize([]) == {}


def test_summarize_counts_not_booleans():
    recs = [{"kind": "choice", "source": "s", "probs": [0.9, 0.1], "gold": 0},
            {"kind": "choice", "source": "s", "probs": [0.2, 0.8], "gold": 0},
            {"kind": "score", "source": "s", "probs": [0.0, 1.0, 0.0], "gold": 1},
            {"kind": "noul", "source": "s", "probs": [0.25, 0.75], "gold": 1}]
    m = summarize(recs)
    assert m["all"]["n"] == 4 and m["kind=choice"]["n"] == 2
    assert abs(m["kind=choice"]["acc"] - 0.5) < 1e-9
    assert abs(m["kind=score"]["score_mae"]) < 1e-9
    assert abs(m["kind=noul"]["noul_brier"] - 0.0625) < 1e-9


def test_temperature_moves_both_ways():
    # overconfident logits -> T > 1 ; underconfident -> T < 1
    over = [[6.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 0.0]]
    golds = [0, 1, 1, 0]  # 25% wrong at ~99.8% confidence
    assert fit_temperature(over, golds) > 1.0
    under = [[0.2, 0.0]] * 20
    assert fit_temperature(under, [0] * 20) < 1.0
