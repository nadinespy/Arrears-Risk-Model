"""Pandera schemas for input and intermediate datasets.

Three schemas:

- ``RawHouseholdSchema`` — what comes out of ``data/household_data.xlsx``.
- ``RawIMDSchema`` — what comes out of the ``IMD25`` sheet of the MHCLG
  Indices of Deprivation 2025 file.
- ``JoinedSchema`` — household rows after the IMD join and binary recoding
  (``Yes``/``No`` → ``1``/``0``). This is the canonical input to feature
  engineering.

Schemas use ``strict=True`` so unexpected columns and missing required
columns both raise: typos and silent column drops surface immediately at
the data-loading boundary rather than later in feature engineering.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

# IMD25 column headers as they actually appear in the MHCLG file. The
# decile header is truncated upstream (missing closing parenthesis) — we
# preserve the literal so pandera's strict matching works without
# pre-renaming. Loaders rename these to canonical short names before
# downstream consumers see them.
IMD_LSOA_CODE_COL = "LSOA code (2021)"
IMD_LSOA_NAME_COL = "LSOA name (2021)"
IMD_LAD_CODE_COL = "Local Authority District code (2024)"
IMD_LAD_NAME_COL = "Local Authority District name (2024)"
IMD_RANK_COL = "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)"
IMD_DECILE_COL = "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA"

# Canonical category sets observed in the household data. Used by the
# raw schema as ``Check.isin`` to catch unexpected values (e.g. a stray
# typo or a new category that arrives in a later data drop and would
# silently break one-hot encoding).
WARDS = (
    "Bellingham", "Blackheath", "Brockley", "Catford South", "Crofton Park",
    "Deptford", "Downham", "Evelyn", "Forest Hill", "Grove Park",
    "Hither Green", "Ladywell", "Lee Green", "Lewisham Central",
    "New Cross Gate", "Perry Vale", "Rushey Green", "Sydenham", "Telegraph Hill",
)
HOUSEHOLD_TYPES = ("Couple with children", "Couple without children", "Lone parent", "Single")
AGE_BRACKETS = (
    "16-17", "18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89",
    "90-94", "95-99", "100+",
)
TENURE_TYPES = (
    "Council tenant", "Owner-occupier", "Private tenant",
    "Social tenant", "Supported housing", "Temporary accommodation",
)
ECONOMIC_STATUSES = ("In work", "Not in work")
DISABILITY_LABELS = ("Disabled", "Not disabled")
YES_NO = ("Yes", "No")

# Columns where the raw value is a "Yes"/"No" string and is recoded to a
# 0/1 int by ``data.recode_binaries``. ``disability`` is a special case
# (label is "Disabled"/"Not disabled") and is handled separately.
BINARY_YES_NO_COLUMNS = (
    "universal_credit", "fuel_poverty", "food_poverty", "arrears_flag",
    "ctc_takeup", "whd_takeup", "fsm_eligible",
)


def _yes_no_column() -> Column:
    return Column(str, Check.isin(YES_NO))


def _zero_one_column() -> Column:
    return Column(int, Check.isin((0, 1)))


RawHouseholdSchema: DataFrameSchema = DataFrameSchema(
    columns={
        "reference": Column(str, unique=True, nullable=False),
        # LSOA 2021 codes for English LSOAs all start with "E01" followed
        # by six digits. Anchor the regex to catch garbage like Excel
        # rendering an LSOA code as a number or trimming a leading zero.
        "lsoa21cd": Column(str, Check.str_matches(r"^E01\d{6}$")),
        "ward": Column(str, Check.isin(WARDS)),
        "household_type": Column(str, Check.isin(HOUSEHOLD_TYPES)),
        "age_bracket": Column(str, Check.isin(AGE_BRACKETS)),
        "tenure_type": Column(str, Check.isin(TENURE_TYPES)),
        "economic_status": Column(str, Check.isin(ECONOMIC_STATUSES)),
        "disability": Column(str, Check.isin(DISABILITY_LABELS)),
        # Numeric ranges deliberately loose: schema's job is to catch
        # type/category surprises, not to enforce business rules. The
        # real data contains a small number of negative monthly_rent
        # values (refunds/corrections), which we want to preserve and
        # let downstream code decide how to handle.
        "monthly_rent": Column(float),
        "lha_shortfall_weekly": Column(float, Check.ge(0)),
        # One NaN observed in the real data; nullable=True keeps it.
        "ben_cap_amount": Column(float, Check.ge(0), nullable=True),
        "universal_credit": _yes_no_column(),
        "income_after_costs": Column(float),
        "fuel_poverty": _yes_no_column(),
        "food_poverty": _yes_no_column(),
        "overpayment_balance": Column(float, Check.ge(0)),
        "arrears_amount": Column(float, Check.ge(0)),
        "arrears_flag": _yes_no_column(),
        "ctc_takeup": _yes_no_column(),
        "whd_takeup": _yes_no_column(),
        "fsm_eligible": _yes_no_column(),
    },
    strict=True,
    coerce=False,
)


RawIMDSchema: DataFrameSchema = DataFrameSchema(
    columns={
        IMD_LSOA_CODE_COL: Column(str, Check.str_matches(r"^E01\d{6}$"), unique=True),
        IMD_LSOA_NAME_COL: Column(str),
        IMD_LAD_CODE_COL: Column(str, Check.str_matches(r"^E0[6-9]\d{6}$")),
        IMD_LAD_NAME_COL: Column(str),
        # 33,755 English LSOAs in IMD2025; rank is 1..33755 with 1 = most deprived.
        IMD_RANK_COL: Column(int, Check.in_range(1, 33755)),
        IMD_DECILE_COL: Column(int, Check.in_range(1, 10)),
    },
    strict=True,
    coerce=False,
)


# Joined dataset: household rows after IMD attach and Yes/No recoding.
# Categorical text columns (ward, household_type, etc.) stay as strings
# until feature engineering encodes them.
JoinedSchema: DataFrameSchema = DataFrameSchema(
    columns={
        "reference": Column(str, unique=True),
        "lsoa21cd": Column(str, Check.str_matches(r"^E01\d{6}$")),
        "ward": Column(str, Check.isin(WARDS)),
        "household_type": Column(str, Check.isin(HOUSEHOLD_TYPES)),
        "age_bracket": Column(str, Check.isin(AGE_BRACKETS)),
        "tenure_type": Column(str, Check.isin(TENURE_TYPES)),
        "economic_status": Column(str, Check.isin(ECONOMIC_STATUSES)),
        "disability": _zero_one_column(),
        "monthly_rent": Column(float),
        "lha_shortfall_weekly": Column(float, Check.ge(0)),
        "ben_cap_amount": Column(float, Check.ge(0), nullable=True),
        "universal_credit": _zero_one_column(),
        "income_after_costs": Column(float),
        "fuel_poverty": _zero_one_column(),
        "food_poverty": _zero_one_column(),
        "overpayment_balance": Column(float, Check.ge(0)),
        "arrears_amount": Column(float, Check.ge(0)),
        "arrears_flag": _zero_one_column(),
        "ctc_takeup": _zero_one_column(),
        "whd_takeup": _zero_one_column(),
        "fsm_eligible": _zero_one_column(),
        "imd_rank": Column(int, Check.in_range(1, 33755)),
        "imd_decile": Column(int, Check.in_range(1, 10)),
    },
    strict=True,
    coerce=False,
)


__all__ = [
    "AGE_BRACKETS",
    "BINARY_YES_NO_COLUMNS",
    "DISABILITY_LABELS",
    "ECONOMIC_STATUSES",
    "HOUSEHOLD_TYPES",
    "IMD_DECILE_COL",
    "IMD_LAD_CODE_COL",
    "IMD_LAD_NAME_COL",
    "IMD_LSOA_CODE_COL",
    "IMD_LSOA_NAME_COL",
    "IMD_RANK_COL",
    "TENURE_TYPES",
    "WARDS",
    "YES_NO",
    "JoinedSchema",
    "RawHouseholdSchema",
    "RawIMDSchema",
]
