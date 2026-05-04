"""Tests for pandera schemas."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from arrears_risk_model.schemas import (
    IMD_DECILE_COL,
    JoinedSchema,
    RawHouseholdSchema,
    RawIMDSchema,
)


def test_raw_household_schema_validates_fixture(raw_household_df: pd.DataFrame) -> None:
    """The synthetic household fixture passes its schema unchanged."""
    validated = RawHouseholdSchema.validate(raw_household_df)
    assert len(validated) == len(raw_household_df)


def test_raw_imd_schema_validates_fixture(raw_imd_df: pd.DataFrame) -> None:
    """The synthetic IMD fixture passes its schema unchanged."""
    validated = RawIMDSchema.validate(raw_imd_df)
    assert len(validated) == len(raw_imd_df)


def test_joined_schema_validates_fixture(joined_df: pd.DataFrame) -> None:
    """The post-join post-recode fixture passes ``JoinedSchema``."""
    validated = JoinedSchema.validate(joined_df)
    assert len(validated) == len(joined_df)


def test_raw_household_rejects_missing_required_column(raw_household_df: pd.DataFrame) -> None:
    """Dropping a required column triggers a SchemaError, not a silent pass."""
    bad = raw_household_df.drop(columns=["arrears_flag"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawHouseholdSchema.validate(bad)


def test_raw_household_rejects_unexpected_column(raw_household_df: pd.DataFrame) -> None:
    """An extra column trips ``strict=True`` — protects against schema drift."""
    bad = raw_household_df.assign(unexpected_col="x")
    with pytest.raises((SchemaError, SchemaErrors)):
        RawHouseholdSchema.validate(bad)


def test_raw_household_rejects_unknown_category(raw_household_df: pd.DataFrame) -> None:
    """A new tenure_type value (e.g. typo or new category) is rejected."""
    bad = raw_household_df.copy()
    bad.loc[0, "tenure_type"] = "Mansion"  # not in TENURE_TYPES
    with pytest.raises((SchemaError, SchemaErrors)):
        RawHouseholdSchema.validate(bad)


def test_raw_household_rejects_wrong_dtype(raw_household_df: pd.DataFrame) -> None:
    """Numeric column delivered as object dtype is rejected."""
    bad = raw_household_df.copy()
    bad["monthly_rent"] = bad["monthly_rent"].astype(str)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawHouseholdSchema.validate(bad)


def test_raw_household_rejects_malformed_lsoa_code(raw_household_df: pd.DataFrame) -> None:
    """A garbled LSOA code (wrong prefix or non-numeric tail) is rejected."""
    bad = raw_household_df.copy()
    bad.loc[0, "lsoa21cd"] = "W01000001"  # Welsh prefix — not English
    with pytest.raises((SchemaError, SchemaErrors)):
        RawHouseholdSchema.validate(bad)


def test_raw_household_rejects_negative_arrears(raw_household_df: pd.DataFrame) -> None:
    """``arrears_amount`` is enforced ≥ 0; negative arrears shouldn't exist."""
    bad = raw_household_df.copy()
    bad.loc[0, "arrears_amount"] = -10.0
    with pytest.raises((SchemaError, SchemaErrors)):
        RawHouseholdSchema.validate(bad)


def test_raw_imd_rejects_out_of_range_decile(raw_imd_df: pd.DataFrame) -> None:
    """Decile must be in 1..10; an 11 is a clear data corruption."""
    bad = raw_imd_df.copy()
    bad.loc[0, IMD_DECILE_COL] = 11
    with pytest.raises((SchemaError, SchemaErrors)):
        RawIMDSchema.validate(bad)


def test_raw_imd_rejects_out_of_range_rank(raw_imd_df: pd.DataFrame) -> None:
    """Rank must be in 1..33,755 (number of English LSOAs)."""
    bad = raw_imd_df.copy()
    bad.loc[0, "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)"] = 0
    with pytest.raises((SchemaError, SchemaErrors)):
        RawIMDSchema.validate(bad)


def test_joined_rejects_unrecoded_binary(joined_df: pd.DataFrame) -> None:
    """A binary column left as ``Yes``/``No`` is caught — the recode step is required."""
    bad = joined_df.copy()
    bad["arrears_flag"] = bad["arrears_flag"].map({0: "No", 1: "Yes"})
    with pytest.raises((SchemaError, SchemaErrors)):
        JoinedSchema.validate(bad)


def test_joined_rejects_missing_imd_attach(joined_df: pd.DataFrame) -> None:
    """Dropping the joined IMD columns is caught — protects against silent join failure."""
    bad = joined_df.drop(columns=["imd_rank", "imd_decile"])
    with pytest.raises((SchemaError, SchemaErrors)):
        JoinedSchema.validate(bad)
