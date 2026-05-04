"""Data loading, joining, and recoding for the arrears risk model.

Pipeline shape::

    load_household_data(path) ─┐
                               ├─→ join_imd ─→ recode_binaries ─→ JoinedSchema
    load_imd(path) ────────────┘

Each function is small and validates its output via a pandera schema, so
a malformed input file or a silent column drop fails at the boundary
where it occurs rather than reaching feature engineering. ``prepare_dataset``
is a convenience that runs the full chain and is what the training and
prediction CLIs will call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arrears_risk_model.logging_config import get_logger
from arrears_risk_model.schemas import (
    BINARY_YES_NO_COLUMNS,
    IMD_DECILE_COL,
    IMD_LSOA_CODE_COL,
    IMD_RANK_COL,
    JoinedSchema,
    RawHouseholdSchema,
    RawIMDSchema,
)

logger = get_logger(__name__)

IMD_SHEET_NAME = "IMD25"

# Mapping from the literal MHCLG headers to canonical short names used
# downstream. Applied in ``join_imd`` after the raw IMD frame validates.
_IMD_RENAME_MAP = {
    IMD_LSOA_CODE_COL: "lsoa21cd",
    IMD_RANK_COL: "imd_rank",
    IMD_DECILE_COL: "imd_decile",
}


def load_household_data(path: Path | str) -> pd.DataFrame:
    """Read and validate ``household_data.xlsx``.

    The input must conform to :data:`RawHouseholdSchema`; the function
    raises ``pandera.errors.SchemaError`` on any mismatch (unknown
    category, wrong dtype, missing/extra column, malformed LSOA code).
    """
    path = Path(path)
    logger.info("Reading household data from %s", path)
    df = pd.read_excel(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    null_counts = df.isna().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        logger.info("Null counts: %s", nulls.to_dict())

    return RawHouseholdSchema.validate(df)


def load_imd(path: Path | str) -> pd.DataFrame:
    """Read and validate the ``IMD25`` sheet of the MHCLG indices file."""
    path = Path(path)
    logger.info("Reading IMD data from %s (sheet=%s)", path, IMD_SHEET_NAME)
    df = pd.read_excel(path, sheet_name=IMD_SHEET_NAME)
    logger.info("Loaded %d LSOAs", len(df))
    return RawIMDSchema.validate(df)


def join_imd(household_df: pd.DataFrame, imd_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join IMD rank and decile onto household rows by ``lsoa21cd``.

    Drops the LAD/LSOA name columns (not used downstream) and renames
    the long MHCLG headers to canonical short names. Raises if any
    household LSOA fails to match — a missing match is a data integrity
    issue, not something to silently fill.
    """
    imd_subset = (
        imd_df[[IMD_LSOA_CODE_COL, IMD_RANK_COL, IMD_DECILE_COL]]
        .rename(columns=_IMD_RENAME_MAP)
    )

    n_before = len(household_df)
    joined = household_df.merge(imd_subset, on="lsoa21cd", how="left", validate="many_to_one")

    if len(joined) != n_before:
        # ``validate="many_to_one"`` should already prevent this; the assert
        # is a belt-and-braces check in case the validation contract changes.
        raise RuntimeError(
            f"Row count changed during IMD join: {n_before} → {len(joined)}"
        )

    unmatched = joined["imd_rank"].isna().sum()
    if unmatched:
        sample = joined.loc[joined["imd_rank"].isna(), "lsoa21cd"].unique()[:5].tolist()
        raise ValueError(
            f"{unmatched} household rows have no matching LSOA in IMD "
            f"(sample codes: {sample}). Cannot proceed without IMD attach."
        )

    logger.info("IMD join: %d rows, all LSOAs matched", len(joined))
    return joined


def recode_binaries(df: pd.DataFrame) -> pd.DataFrame:
    """Recode ``Yes``/``No`` and ``Disabled``/``Not disabled`` to ``1``/``0``.

    Returns a new frame; the input is not mutated. Raises if a column
    contains an unexpected value (e.g. ``"yes"`` lowercase or ``"Y"``)
    rather than silently producing NaN.
    """
    out = df.copy()

    yes_no_map = {"Yes": 1, "No": 0}
    for col in BINARY_YES_NO_COLUMNS:
        if col not in out.columns:
            continue
        unknown = set(out[col].dropna().unique()) - set(yes_no_map)
        if unknown:
            raise ValueError(f"Column {col!r} has unexpected values: {sorted(unknown)}")
        out[col] = out[col].map(yes_no_map).astype(int)

    if "disability" in out.columns:
        disability_map = {"Disabled": 1, "Not disabled": 0}
        unknown = set(out["disability"].dropna().unique()) - set(disability_map)
        if unknown:
            raise ValueError(f"Column 'disability' has unexpected values: {sorted(unknown)}")
        out["disability"] = out["disability"].map(disability_map).astype(int)

    return out


def verify_arrears_consistency(df: pd.DataFrame) -> int:
    """Cross-check ``arrears_flag`` against ``arrears_amount``.

    A consistent row has ``arrears_flag == 'Yes'`` exactly when
    ``arrears_amount > 0``. Logs a warning with the count and returns
    it; does not raise, because small inconsistencies (rounding,
    in-flight payments) can legitimately appear in council data and
    we want them surfaced rather than silently filtered.
    """
    flag_col = df["arrears_flag"]
    # Works on both raw ("Yes"/"No" strings) and recoded (0/1 ints) frames.
    # ``dtype == object`` would miss pandas 3's PyArrow-backed string dtype.
    if pd.api.types.is_numeric_dtype(flag_col):
        flag_positive = flag_col.astype(bool)
    else:
        flag_positive = flag_col.eq("Yes")

    amount_positive = df["arrears_amount"] > 0
    inconsistent = (flag_positive != amount_positive).sum()

    if inconsistent:
        logger.warning(
            "%d/%d rows have inconsistent arrears_flag vs arrears_amount",
            inconsistent, len(df),
        )
    else:
        logger.info("Arrears flag/amount consistency check: all %d rows OK", len(df))

    return int(inconsistent)


def prepare_dataset(household_path: Path | str, imd_path: Path | str) -> pd.DataFrame:
    """Run the full load → join → recode chain. Returns a frame that
    validates against :data:`JoinedSchema`.
    """
    household = load_household_data(household_path)
    imd = load_imd(imd_path)
    verify_arrears_consistency(household)
    joined = join_imd(household, imd)
    recoded = recode_binaries(joined)
    return JoinedSchema.validate(recoded)


__all__ = [
    "join_imd",
    "load_household_data",
    "load_imd",
    "prepare_dataset",
    "recode_binaries",
    "verify_arrears_consistency",
]
