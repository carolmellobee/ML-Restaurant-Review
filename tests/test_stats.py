"""Fast checks on the statistical building blocks. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.experiment.analysis import compare_proportions, srm_check
from src.experiment.power import mde_given_n, sample_size_two_proportions, simulated_power
from src.experiment.traps import obf_constant
from src.nlp.complaints import tag


def test_sample_size_matches_statsmodels_within_3pct():
    ours = sample_size_two_proportions(0.18, 0.05)
    h = proportion_effectsize(0.18 * 1.05, 0.18)
    ref = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8)
    assert abs(ours / ref - 1) < 0.03


def test_formula_power_is_confirmed_by_simulation():
    n = sample_size_two_proportions(0.18, 0.05)
    assert simulated_power(0.18, 0.05, n, sims=6000) == pytest.approx(0.80, abs=0.03)


def test_mde_inverts_sample_size():
    n = sample_size_two_proportions(0.18, 0.05)
    assert mde_given_n(0.18, n) == pytest.approx(0.05, rel=0.01)


def test_obf_constant_for_four_looks():
    # published value for K=4, two-sided alpha=0.05 is ~2.024
    assert obf_constant(4) == pytest.approx(2.024, abs=0.02)


def test_aa_confidence_interval_coverage():
    rng = np.random.default_rng(0)
    covered = 0
    for _ in range(300):
        n = 4000
        df = pd.DataFrame({"variant": ["control"] * n + ["treatment"] * n,
                           "y": rng.binomial(1, 0.2, 2 * n)})
        lo, hi = compare_proportions(df, "y")["abs_ci"]
        covered += lo <= 0 <= hi
    assert covered / 300 == pytest.approx(0.95, abs=0.03)


def test_srm_flags_a_52_48_split():
    df = pd.DataFrame({"variant": ["control"] * 52_000 + ["treatment"] * 48_000})
    assert srm_check(df)["srm_detected"]


def test_srm_passes_a_fair_split():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"variant": np.where(rng.random(100_000) < 0.5, "control", "treatment")})
    assert not srm_check(df)["srm_detected"]


@pytest.mark.parametrize("text,cat,expected", [
    ("Very unfriendly attitude from the staff", "friendly_staff", 0),
    ("Very unfriendly attitude from the staff", "rude_staff", 1),
    ("The staff was super friendly", "friendly_staff", 1),
    ("Delivery was not fast at all", "fast_delivery", 0),
    ("Found a hair in my ramen", "hygiene", 1),
    ("The ETA kept going up", "delivery_delay", 1),
])
def test_lexicon_edge_cases(text, cat, expected):
    assert tag(pd.Series([text]))[cat].iloc[0] == expected
