"""Tests for the data loading and joining module."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from arrears_risk_model.data import (
    join_imd,
    load_household_data,
    load_imd,
    prepare_dataset,
    recode_binaries,
    verify_arrears_consistency,
)
from arrears_risk_model.schemas import (
    BINARY_YES_NO_COLUMNS,
    IMD_DECILE_COL,
    IMD_RANK_COL,
)


@pytest.fixture
def household_xlsx(raw_household_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write the household fixture to disk so loaders can be exercised end-to-end."""
    path = tmp_path / "household.xlsx"
    raw_household_df.to_excel(path, index=False)
    return path


@pytest.fixture
def imd_xlsx(raw_imd_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write the IMD fixture to disk on the expected ``IMD25`` sheet."""
    path = tmp_path / "imd.xlsx"
    with pd.ExcelWriter(path) as writer:
        raw_imd_df.to_excel(writer, sheet_name="IMD25", index=False)
    return path


# ---------- load_household_data ----------

def test_load_household_data_validates(household_xlsx: Path) -> None:
    """Round-trip: write fixture → load → schema validation passes."""
    df = load_household_data(household_xlsx)
    assert len(df) == 50
    assert {"reference", "lsoa21cd", "arrears_flag"}.issubset(df.columns)


def test_load_household_data_rejects_bad_file(
    raw_household_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Schema violations in the source file surface as a SchemaError on load."""
    bad = raw_household_df.assign(unexpected_col="x")
    path = tmp_path / "bad.xlsx"
    bad.to_excel(path, index=False)
    with pytest.raises((SchemaError, SchemaErrors)):
        load_household_data(path)


# ---------- load_imd ----------

def test_load_imd_validates(imd_xlsx: Path) -> None:
    """The IMD25 sheet loads and validates."""
    df = load_imd(imd_xlsx)
    assert IMD_RANK_COL in df.columns
    assert IMD_DECILE_COL in df.columns


# ---------- join_imd ----------

def test_join_imd_attaches_columns(
    raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame
) -> None:
    """Joined frame has imd_rank and imd_decile under canonical names."""
    joined = join_imd(raw_household_df, raw_imd_df)
    assert "imd_rank" in joined.columns
    assert "imd_decile" in joined.columns
    assert joined["imd_rank"].notna().all()
    assert joined["imd_decile"].notna().all()


def test_join_imd_no_row_inflation(
    raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame
) -> None:
    """Many-to-one join must not duplicate household rows."""
    joined = join_imd(raw_household_df, raw_imd_df)
    assert len(joined) == len(raw_household_df)


def test_join_imd_drops_extra_imd_columns(
    raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame
) -> None:
    """LSOA name and LAD code/name aren't carried into the joined frame."""
    joined = join_imd(raw_household_df, raw_imd_df)
    assert "LSOA name (2021)" not in joined.columns
    assert "Local Authority District code (2024)" not in joined.columns


def test_join_imd_raises_on_unmatched_lsoa(
    raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame
) -> None:
    """A household with an LSOA missing from IMD halts the pipeline."""
    truncated_imd = raw_imd_df.iloc[5:].copy()  # drop first few LSOAs the household uses
    with pytest.raises(ValueError, match="no matching LSOA"):
        join_imd(raw_household_df, truncated_imd)


# ---------- recode_binaries ----------

def test_recode_binaries_yes_no(raw_household_df: pd.DataFrame) -> None:
    """Yes/No columns become 0/1 integers."""
    recoded = recode_binaries(raw_household_df)
    for col in BINARY_YES_NO_COLUMNS:
        assert set(recoded[col].unique()).issubset({0, 1})
        assert recoded[col].dtype.kind == "i"


def test_recode_binaries_disability(raw_household_df: pd.DataFrame) -> None:
    """``disability`` (Disabled/Not disabled) recodes to 1/0."""
    recoded = recode_binaries(raw_household_df)
    assert set(recoded["disability"].unique()).issubset({0, 1})
    # Spot-check one row to confirm the mapping direction.
    first_label = raw_household_df["disability"].iloc[0]
    expected = 1 if first_label == "Disabled" else 0
    assert recoded["disability"].iloc[0] == expected


def test_recode_binaries_does_not_mutate_input(raw_household_df: pd.DataFrame) -> None:
    """The original frame is unchanged after recoding."""
    original_dtype = raw_household_df["arrears_flag"].dtype
    _ = recode_binaries(raw_household_df)
    assert raw_household_df["arrears_flag"].dtype == original_dtype
    assert raw_household_df["arrears_flag"].iloc[0] in {"Yes", "No"}


def test_recode_binaries_rejects_unexpected_value(raw_household_df: pd.DataFrame) -> None:
    """A stray ``yes`` lowercase or unknown label is caught before silent NaN."""
    bad = raw_household_df.copy()
    bad.loc[0, "arrears_flag"] = "y"
    with pytest.raises(ValueError, match="unexpected values"):
        recode_binaries(bad)


# ---------- verify_arrears_consistency ----------

def test_verify_arrears_consistency_clean(raw_household_df: pd.DataFrame) -> None:
    """A frame engineered to be consistent reports zero inconsistencies."""
    df = raw_household_df.copy()
    df["arrears_amount"] = df["arrears_flag"].map({"Yes": 100.0, "No": 0.0})
    assert verify_arrears_consistency(df) == 0


def test_verify_arrears_consistency_logs_inconsistencies(
    raw_household_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Mismatches are counted and logged at WARNING level (not raised)."""
    df = raw_household_df.copy()
    df["arrears_amount"] = df["arrears_flag"].map({"Yes": 100.0, "No": 0.0})
    # Flip a "Yes" row's amount to 0 so flag and amount disagree.
    yes_idx = df.index[df["arrears_flag"].eq("Yes")][0]
    df.loc[yes_idx, "arrears_amount"] = 0.0
    with caplog.at_level(logging.WARNING):
        count = verify_arrears_consistency(df)
    assert count == 1
    assert any("inconsistent" in r.message.lower() for r in caplog.records)


# ---------- prepare_dataset ----------

def test_prepare_dataset_e2e(household_xlsx: Path, imd_xlsx: Path) -> None:
    """Full chain: load → consistency check → join → recode → JoinedSchema validation."""
    df = prepare_dataset(household_xlsx, imd_xlsx)
    assert len(df) == 50
    assert {"imd_rank", "imd_decile"}.issubset(df.columns)
    # binaries are recoded
    assert df["arrears_flag"].dtype.kind == "i"
    assert set(df["arrears_flag"].unique()).issubset({0, 1})
