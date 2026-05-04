"""Synthetic fixtures for unit tests.

Real input data is gitignored and may not be present in CI; tests work
off small (~50 row) synthetic frames whose distribution and dtypes
match the production schemas. Fixtures are seeded so failures reproduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arrears_risk_model.schemas import (
    AGE_BRACKETS,
    BINARY_YES_NO_COLUMNS,
    DISABILITY_LABELS,
    ECONOMIC_STATUSES,
    HOUSEHOLD_TYPES,
    IMD_DECILE_COL,
    IMD_LAD_CODE_COL,
    IMD_LAD_NAME_COL,
    IMD_LSOA_CODE_COL,
    IMD_LSOA_NAME_COL,
    IMD_RANK_COL,
    TENURE_TYPES,
    WARDS,
)

N_ROWS = 50
LEWISHAM_LAD_CODE = "E09000023"
LEWISHAM_LAD_NAME = "Lewisham"


def _lsoa_codes(n: int) -> list[str]:
    """Synthetic LSOA codes that pass the ``^E01\\d{6}$`` schema check."""
    return [f"E01{str(3000 + i).zfill(6)}" for i in range(n)]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def raw_household_df(rng: np.random.Generator) -> pd.DataFrame:
    """50 synthetic rows matching ``RawHouseholdSchema``."""
    n = N_ROWS
    lsoas = _lsoa_codes(20)  # 20 distinct LSOAs across 50 households
    df = pd.DataFrame({
        "reference": [f"ID{str(i).zfill(5)}" for i in range(n)],
        "lsoa21cd": rng.choice(lsoas, size=n),
        "ward": rng.choice(WARDS, size=n),
        "household_type": rng.choice(HOUSEHOLD_TYPES, size=n),
        "age_bracket": rng.choice(AGE_BRACKETS, size=n),
        "tenure_type": rng.choice(TENURE_TYPES, size=n),
        "economic_status": rng.choice(ECONOMIC_STATUSES, size=n),
        "disability": rng.choice(DISABILITY_LABELS, size=n),
        "monthly_rent": rng.uniform(400, 2000, size=n),
        "lha_shortfall_weekly": rng.uniform(0, 50, size=n),
        "ben_cap_amount": rng.uniform(0, 100, size=n),
        "universal_credit": rng.choice(("Yes", "No"), size=n),
        "income_after_costs": rng.uniform(200, 2500, size=n),
        "fuel_poverty": rng.choice(("Yes", "No"), size=n),
        "food_poverty": rng.choice(("Yes", "No"), size=n),
        "overpayment_balance": rng.uniform(0, 1000, size=n),
        "arrears_amount": rng.uniform(0, 500, size=n),
        "arrears_flag": rng.choice(("Yes", "No"), size=n, p=[0.25, 0.75]),
        "ctc_takeup": rng.choice(("Yes", "No"), size=n),
        "whd_takeup": rng.choice(("Yes", "No"), size=n),
        "fsm_eligible": rng.choice(("Yes", "No"), size=n),
    })
    # Inject a single NaN in ben_cap_amount to mirror the one observed
    # in the real data, so tests catch any code that assumes no nulls.
    df.loc[0, "ben_cap_amount"] = np.nan
    return df


@pytest.fixture
def raw_imd_df(raw_household_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """IMD frame covering every LSOA appearing in ``raw_household_df``.

    Includes a handful of additional non-Lewisham LSOAs so tests can
    confirm the join filters correctly.
    """
    household_lsoas = sorted(raw_household_df["lsoa21cd"].unique())
    extra_lsoas = [f"E01{str(9000 + i).zfill(6)}" for i in range(5)]  # outside-borough rows
    all_lsoas = household_lsoas + extra_lsoas
    n = len(all_lsoas)
    return pd.DataFrame({
        IMD_LSOA_CODE_COL: all_lsoas,
        IMD_LSOA_NAME_COL: [f"Synth LSOA {i:03d}" for i in range(n)],
        IMD_LAD_CODE_COL: (
            [LEWISHAM_LAD_CODE] * len(household_lsoas)
            + ["E09000022"] * len(extra_lsoas)
        ),
        IMD_LAD_NAME_COL: (
            [LEWISHAM_LAD_NAME] * len(household_lsoas)
            + ["Lambeth"] * len(extra_lsoas)
        ),
        IMD_RANK_COL: rng.integers(low=1, high=33756, size=n).astype(int),
        IMD_DECILE_COL: rng.integers(low=1, high=11, size=n).astype(int),
    })


@pytest.fixture
def joined_df(raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame) -> pd.DataFrame:
    """Expected output of joining household + IMD and recoding binaries.

    Mirrors what ``data.join_imd`` followed by ``data.recode_binaries``
    will produce in Commit 5; reproduced here so the joined-schema test
    has something to validate against.
    """
    imd_short = raw_imd_df.rename(columns={
        IMD_LSOA_CODE_COL: "lsoa21cd",
        IMD_RANK_COL: "imd_rank",
        IMD_DECILE_COL: "imd_decile",
    })[["lsoa21cd", "imd_rank", "imd_decile"]]

    df = raw_household_df.merge(imd_short, on="lsoa21cd", how="left")
    df["disability"] = (df["disability"] == "Disabled").astype(int)
    for col in BINARY_YES_NO_COLUMNS:
        df[col] = (df[col] == "Yes").astype(int)
    df["imd_rank"] = df["imd_rank"].astype(int)
    df["imd_decile"] = df["imd_decile"].astype(int)
    return df
