import os

import numpy as np
import pandas as pd


###############################################################################
# PATHS AND PARAMETERS
###############################################################################

PANEL_PATH = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "5b_out_08052026/combined.parquet"
)

REGISTER_PATH = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "5b_out_04192026/registered.parquet"
)

OUTPUT_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "5c_out_09132026/with_sseg_reg.parquet"
)

PANEL_YEARS = list(range(2018, 2024))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


###############################################################################
# HELPERS
###############################################################################

def section(title: str) -> None:
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    data_name: str,
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{data_name} is missing required columns: "
            f"{missing_columns}"
        )


###############################################################################
# LOAD INPUT DATA
###############################################################################

section("LOADING INPUT DATA")

df_panel = pd.read_parquet(PANEL_PATH)
df_register = pd.read_parquet(REGISTER_PATH)

panel_columns = [
    "contract_account_hashed",
    "wkt",
    "Type",
    "trfname",
    "month_year",
    "shs_area_m2_edit",
    "shs_label_edit",
    "kwh",
    "building_id",
    "n_contracts_building_month",
]

register_columns = [
    "contract_account_hashed",
    "year",
    "date_commissioned",
    "installation_type",
    "total_capacity_(va)",
]

require_columns(
    df_panel,
    panel_columns,
    "Panel data",
)

require_columns(
    df_register,
    register_columns,
    "Registration data",
)

df_panel = df_panel[panel_columns].copy()
df_register = df_register[register_columns].copy()

df_panel["month_year"] = pd.to_datetime(
    df_panel["month_year"],
    errors="coerce",
)

invalid_panel_dates = int(
    df_panel["month_year"].isna().sum()
)

print(
    "Panel rows with invalid month_year:",
    f"{invalid_panel_dates:,}",
    flush=True,
)

if invalid_panel_dates > 0:
    raise ValueError(
        "Panel contains invalid month_year values."
    )

df_panel["year"] = (
    df_panel["month_year"]
    .dt.year
    .astype("Int64")
)

print(
    "Panel rows:",
    f"{len(df_panel):,}",
    flush=True,
)

print(
    "Unique panel accounts:",
    f"{df_panel['contract_account_hashed'].nunique():,}",
    flush=True,
)

input_duplicate_rows = int(
    df_panel.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Input rows involved in duplicate account-months:",
    f"{input_duplicate_rows:,}",
    flush=True,
)

if input_duplicate_rows > 0:
    raise ValueError(
        "Input panel is not unique by account-month."
    )


###############################################################################
# RETAIN M2F SHS STATUS AND CONSTRUCT M2F-ONLY CAPACITY
###############################################################################

section("CONSTRUCTING M2F-ONLY SHS VARIABLES")

# SHS is determined only by the cleaned M2F label from Script 5b.
df_panel["has_shs"] = (
    df_panel["shs_label_edit"]
    .eq("PV_normal")
    .astype("int8")
)

panel_size_m2 = 1.7
watt_per_panel = 400

df_panel["PV_capacity_W_per_contract"] = (
    pd.to_numeric(
        df_panel["shs_area_m2_edit"],
        errors="coerce",
    )
    .where(df_panel["has_shs"].eq(1))
    .replace(0, np.nan)
    * watt_per_panel
    / panel_size_m2
)

# Compatibility alias for downstream scripts.
df_panel["Watt"] = (
    df_panel["PV_capacity_W_per_contract"]
)

df_panel["capacity_source"] = np.where(
    df_panel["PV_capacity_W_per_contract"].notna(),
    "detected_equal_share",
    pd.NA,
)

# Non-M2F registrations must not acquire an SHS label or SHS capacity.
non_shs_with_capacity = (
    df_panel["has_shs"].eq(0)
    & df_panel["PV_capacity_W_per_contract"].notna()
)

print(
    "Non-SHS rows with M2F capacity:",
    f"{non_shs_with_capacity.sum():,}",
    flush=True,
)

if non_shs_with_capacity.any():
    raise ValueError(
        "Some rows without an M2F SHS label have M2F capacity."
    )

m2f_shs_counts = (
    df_panel.loc[df_panel["has_shs"].eq(1)]
    .groupby("year")["contract_account_hashed"]
    .nunique()
    .reindex(PANEL_YEARS, fill_value=0)
)

print(
    "\nUnique M2F SHS accounts by year:",
    flush=True,
)

print(
    m2f_shs_counts,
    flush=True,
)


###############################################################################
# PREPARE OFFICIAL REGISTRATION DATA
###############################################################################

section("PREPARING OFFICIAL REGISTRATION DATA")

# Convert both possible registration-date fields.
df_register["year"] = pd.to_datetime(
    df_register["year"],
    errors="coerce",
)

df_register["date_commissioned"] = pd.to_datetime(
    df_register["date_commissioned"],
    errors="coerce",
)

# Prefer the exact commissioning date.
# Fall back to the year/date field when commissioning date is missing.
df_register["registration_date"] = (
    df_register["date_commissioned"]
    .combine_first(df_register["year"])
)

# Normalize to the first day of the registration month.
df_register["registration_month"] = (
    df_register["registration_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Registration becomes effective in the following calendar month.
df_register["registration_effective_month"] = (
    df_register["registration_month"]
    + pd.offsets.MonthBegin(1)
)

df_register["registration_year"] = (
    df_register["registration_month"]
    .dt.year
    .astype("Int64")
)

n_from_commissioned = int(
    df_register["date_commissioned"].notna().sum()
)

n_from_fallback_year = int(
    (
        df_register["date_commissioned"].isna()
        & df_register["year"].notna()
    ).sum()
)

n_missing_registration_date = int(
    df_register["registration_date"].isna().sum()
)

print(
    "Registration dates from date_commissioned:",
    f"{n_from_commissioned:,}",
    flush=True,
)

print(
    "Registration dates from fallback year field:",
    f"{n_from_fallback_year:,}",
    flush=True,
)

print(
    "Registration records with no usable date:",
    f"{n_missing_registration_date:,}",
    flush=True,
)

df_register = df_register.drop(
    columns=[
        "year",
        "date_commissioned",
    ]
)

df_register["registered_capacity_va"] = pd.to_numeric(
    df_register["total_capacity_(va)"],
    errors="coerce",
)

df_register = df_register.drop(
    columns=["total_capacity_(va)"]
)

print(
    "Unique accounts in registration file:",
    f"{df_register['contract_account_hashed'].nunique():,}",
    flush=True,
)

print(
    "Registration rows with missing account:",
    f"{df_register['contract_account_hashed'].isna().sum():,}",
    flush=True,
)

print(
    "\nRegistration-year distribution:",
    flush=True,
)

print(
    df_register["registration_year"]
    .value_counts(dropna=False)
    .sort_index(),
    flush=True,
)

registration_key = [
    "contract_account_hashed",
    "registration_month",
]

duplicate_registration_rows = (
    df_register.duplicated(
        subset=registration_key,
        keep=False,
    )
)

print(
    "\nRows involved in duplicate account-registration-month keys:",
    f"{duplicate_registration_rows.sum():,}",
    flush=True,
)

print(
    "Distinct duplicated account-registration-month keys:",
    df_register.loc[
        duplicate_registration_rows,
        registration_key,
    ].drop_duplicates().shape[0],
    flush=True,
)

if duplicate_registration_rows.any():
    duplicate_detail = df_register.loc[
        duplicate_registration_rows
    ].copy()

    conflict_counts = {}

    for column in [
        "installation_type",
        "registered_capacity_va",
    ]:
        values_per_key = (
            duplicate_detail
            .groupby(
                registration_key,
                dropna=False,
            )[column]
            .nunique(dropna=False)
        )

        conflict_counts[column] = int(
            values_per_key.gt(1).sum()
        )

    print(
        "Conflicting duplicate registration keys:",
        flush=True,
    )

    print(
        pd.Series(conflict_counts),
        flush=True,
    )

    if any(
        count > 0
        for count in conflict_counts.values()
    ):
        print(
            duplicate_detail
            .sort_values(registration_key)
            .head(50)
            .to_string(),
            flush=True,
        )

        raise ValueError(
            "Registration data contain conflicting "
            "duplicate account-month records."
        )

    df_register = (
        df_register
        .drop_duplicates(
            subset=registration_key,
            keep="first",
        )
        .copy()
    )

remaining_registration_duplicates = int(
    df_register.duplicated(
        subset=registration_key,
        keep=False,
    ).sum()
)

print(
    "Remaining duplicate registration rows:",
    f"{remaining_registration_duplicates:,}",
    flush=True,
)

if remaining_registration_duplicates > 0:
    raise ValueError(
        "Registration lookup remains nonunique."
    )


###############################################################################
# MATCH-RATE AND TIMING DIAGNOSTICS
###############################################################################

section("REGISTRATION MATCH-RATE DIAGNOSTICS")

panel_accounts = set(
    df_panel["contract_account_hashed"]
    .dropna()
    .unique()
)

register_accounts = set(
    df_register["contract_account_hashed"]
    .dropna()
    .unique()
)

matched_accounts = (
    panel_accounts
    & register_accounts
)

print(
    "Panel accounts:",
    f"{len(panel_accounts):,}",
    flush=True,
)

print(
    "Registered accounts:",
    f"{len(register_accounts):,}",
    flush=True,
)

print(
    "Accounts in both sources:",
    f"{len(matched_accounts):,}",
    flush=True,
)

print(
    "Accounts only in registration file:",
    f"{len(register_accounts - panel_accounts):,}",
    flush=True,
)

registration_match_rate = (
    100
    * len(matched_accounts)
    / max(len(register_accounts), 1)
)

print(
    "Registration-account match rate:",
    f"{registration_match_rate:.2f}%",
    flush=True,
)

registration_account_summary = (
    df_register
    .sort_values(
        [
            "contract_account_hashed",
            "registration_year",
        ]
    )
    .groupby(
        "contract_account_hashed",
        as_index=False,
    )
    .agg(
        first_registration_month=(
            "registration_month",
            "min",
        ),
        last_registration_month=(
            "registration_month",
            "max",
        ),
        n_registration_months=(
            "registration_month",
            "nunique",
        ),
    )
)

registration_account_summary["first_registration_year"] = (
    registration_account_summary["first_registration_month"]
    .dt.year
    .astype("Int64")
)

registration_account_summary["last_registration_year"] = (
    registration_account_summary["last_registration_month"]
    .dt.year
    .astype("Int64")
)

registration_account_summary["in_panel"] = (
    registration_account_summary[
        "contract_account_hashed"
    ].isin(panel_accounts)
)

print(
    "\nRegistration accounts by first registration year "
    "and panel match:",
    flush=True,
)

print(
    registration_account_summary
    .groupby(
        [
            "first_registration_year",
            "in_panel",
        ],
        dropna=False,
    )["contract_account_hashed"]
    .nunique()
    .unstack(fill_value=0),
    flush=True,
)

matched_registration_accounts = (
    registration_account_summary.loc[
        registration_account_summary["in_panel"]
    ]
    .copy()
)

matched_by_2023 = int(
    matched_registration_accounts[
        "first_registration_year"
    ]
    .le(2023)
    .sum()
)

matched_after_2023 = int(
    matched_registration_accounts[
        "first_registration_year"
    ]
    .gt(2023)
    .sum()
)

matched_missing_year = int(
    matched_registration_accounts[
        "first_registration_year"
    ]
    .isna()
    .sum()
)

print(
    "\nMatched registration accounts with first "
    "registration by 2023:",
    f"{matched_by_2023:,}",
    flush=True,
)

print(
    "Matched registration accounts with first "
    "registration after 2023:",
    f"{matched_after_2023:,}",
    flush=True,
)

print(
    "Matched registration accounts with missing "
    "registration year:",
    f"{matched_missing_year:,}",
    flush=True,
)


###############################################################################
# BUILD ABSORBING REGISTRATION STATUS
###############################################################################

section("CREATING ABSORBING REGISTRATION STATUS")

# Use the earliest known registration month for each account.
# Sorting first ensures installation type and registered capacity come from
# the earliest registration record.
registration_first = (
    df_register
    .sort_values(
        [
            "contract_account_hashed",
            "registration_month",
        ]
    )
    .drop_duplicates(
        subset=["contract_account_hashed"],
        keep="first",
    )
    [
        [
            "contract_account_hashed",
            "registration_date",
            "registration_month",
            "registration_effective_month",
            "registration_year",
            "installation_type",
            "registered_capacity_va",
        ]
    ]
    .copy()
)

duplicate_first_registration = int(
    registration_first.duplicated(
        subset=["contract_account_hashed"],
        keep=False,
    ).sum()
)

if duplicate_first_registration > 0:
    raise ValueError(
        "First-registration lookup is not unique by account."
    )

rows_before_merge = len(df_panel)

df_clean = df_panel.merge(
    registration_first,
    on="contract_account_hashed",
    how="left",
    validate="many_to_one",
)

print(
    "Rows before registration merge:",
    f"{rows_before_merge:,}",
    flush=True,
)

print(
    "Rows after registration merge:",
    f"{len(df_clean):,}",
    flush=True,
)

if len(df_clean) != rows_before_merge:
    raise ValueError(
        "Registration merge changed the panel row count."
    )

# Normalize each panel observation to its calendar month.
df_clean["panel_month"] = (
    df_clean["month_year"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Registration turns on in the calendar month AFTER the registration month
# and remains on for all later panel months.
df_clean["has_registration"] = (
    df_clean["registration_effective_month"].notna()
    & df_clean["panel_month"].ge(
        df_clean["registration_effective_month"]
    )
).astype("int8")

###############################################################################
# CREATE MONTHLY REGISTRATION STATUS
###############################################################################

df_clean["panel_month"] = (
    df_clean["month_year"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Registration begins in the month after commissioning/registration.
df_clean["has_registration"] = (
    df_clean["registration_effective_month"].notna()
    & df_clean["panel_month"].ge(
        df_clean["registration_effective_month"]
    )
).astype("int8")

###############################################################################
# REGISTRATION TIMING AUDIT
###############################################################################

registration_timing_audit = (
    df_clean
    .groupby(
        "contract_account_hashed",
        as_index=False,
    )
    .agg(
        first_panel_month=(
            "panel_month",
            "min",
        ),
        last_panel_month=(
            "panel_month",
            "max",
        ),
        registration_month=(
            "registration_month",
            "first",
        ),
        registration_effective_month=(
            "registration_effective_month",
            "first",
        ),
        ever_registered=(
            "has_registration",
            "max",
        ),
    )
)

matched_by_2023_audit = registration_timing_audit.loc[
    registration_timing_audit[
        "registration_month"
    ].notna()
    & registration_timing_audit[
        "registration_month"
    ].dt.year.le(2023)
].copy()

matched_by_2023_audit["reason_not_registered"] = np.select(
    [
        matched_by_2023_audit["ever_registered"].eq(1),

        matched_by_2023_audit[
            "registration_effective_month"
        ].isna(),

        matched_by_2023_audit[
            "last_panel_month"
        ].lt(
            matched_by_2023_audit[
                "registration_effective_month"
            ]
        ),
    ],
    [
        "registered_in_panel",
        "missing_registration_month",
        "panel_ends_before_effective_month",
    ],
    default="other",
)

print(
    "\nRegistration timing audit:"
)

print(
    matched_by_2023_audit[
        "reason_not_registered"
    ].value_counts(dropna=False)
)

# Registration details should appear only once registration is effective.
before_registration = (
    df_clean["has_registration"].eq(0)
)

df_clean.loc[
    before_registration,
    [
        "installation_type",
        "registered_capacity_va",
    ],
] = [
    pd.NA,
    np.nan,
]

# Confirm the absorbing-status rule.
registration_reversals = (
    df_clean
    .sort_values(
        [
            "contract_account_hashed",
            "month_year",
        ]
    )
    .groupby(
        "contract_account_hashed"
    )["has_registration"]
    .diff()
    .lt(0)
    .sum()
)

print(
    "Registration reversals from 1 back to 0:",
    f"{registration_reversals:,}",
    flush=True,
)

effective_month_examples = (
    df_clean.loc[
        df_clean["registration_effective_month"].notna(),
        [
            "contract_account_hashed",
            "registration_month",
            "registration_effective_month",
        ],
    ]
    .drop_duplicates()
    .sort_values(
        [
            "registration_month",
            "contract_account_hashed",
        ]
    )
    .head(20)
)

print(
    "\nExample registration and effective months:",
    flush=True,
)

print(
    effective_month_examples.to_string(index=False),
    flush=True,
)

incorrect_early_flags = (
    df_clean["has_registration"].eq(1)
    & df_clean["panel_month"].lt(
        df_clean["registration_effective_month"]
    )
)

print(
    "Rows incorrectly flagged before effective month:",
    f"{incorrect_early_flags.sum():,}",
    flush=True,
)

if incorrect_early_flags.any():
    raise ValueError(
        "Some observations are registered before the month "
        "after registration."
    )

if registration_reversals > 0:
    raise ValueError(
        "Registration status is not absorbing."
    )

registered_by_year = (
    df_clean.loc[
        df_clean["has_registration"].eq(1)
    ]
    .groupby("year")[
        "contract_account_hashed"
    ]
    .nunique()
    .reindex(
        PANEL_YEARS,
        fill_value=0,
    )
    .rename("registered_contracts")
    .reset_index()
)

print(
    "\nRegistered accounts represented in each panel year:",
    flush=True,
)

print(
    registered_by_year,
    flush=True,
)

unique_registered_contracts = (
    df_clean.loc[
        df_clean["has_registration"].eq(1),
        "contract_account_hashed",
    ]
    .nunique()
)

print(
    "Unique accounts ever registered during panel:",
    f"{unique_registered_contracts:,}",
    flush=True,
)


###############################################################################
# VERIFY REGISTRATION DOES NOT ALTER SHS
###############################################################################

section("VERIFYING REGISTRATION DOES NOT ALTER M2F SHS")

registered_without_m2f = (
    df_clean["has_registration"].eq(1)
    & df_clean["has_shs"].eq(0)
)

print(
    "Registered rows without an M2F SHS detection:",
    f"{registered_without_m2f.sum():,}",
    flush=True,
)

n_registered_without_m2f = (
    df_clean.loc[
        registered_without_m2f,
        "contract_account_hashed",
    ]
    .nunique()
)

print(
    "Unique registered accounts without an M2F SHS detection in the same row:",
    f"{n_registered_without_m2f:,}",
    flush=True,
)

# These rows must retain missing SHS capacity.
registered_non_m2f_with_capacity = (
    registered_without_m2f
    & df_clean["PV_capacity_W_per_contract"].notna()
)

print(
    "Registered non-M2F rows incorrectly assigned "
    "SHS capacity:",
    f"{registered_non_m2f_with_capacity.sum():,}",
    flush=True,
)

if registered_non_m2f_with_capacity.any():
    raise ValueError(
        "Registration incorrectly created M2F SHS capacity."
    )

# Official registration capacity is separate from M2F capacity.
print(
    "\nCapacity variables retained separately:",
    flush=True,
)

print(
    [
        "PV_capacity_W_per_contract",
        "registered_capacity_va",
    ],
    flush=True,
)


###############################################################################
# FINAL QA
###############################################################################

section("FINAL QA")

status_consistency = (
    df_clean
    .groupby("year")
    .agg(
        has_shs_rows=(
            "has_shs",
            "sum",
        ),
        pv_label_rows=(
            "shs_label_edit",
            lambda values: (
                values.eq("PV_normal").sum()
            ),
        ),
    )
)

status_consistency["difference"] = (
    status_consistency["has_shs_rows"]
    - status_consistency["pv_label_rows"]
)

print(
    "M2F binary-label consistency:",
    flush=True,
)

print(
    status_consistency,
    flush=True,
)

if status_consistency["difference"].ne(0).any():
    raise ValueError(
        "has_shs and shs_label_edit are inconsistent."
    )

final_duplicate_rows = int(
    df_clean.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Final duplicate account-month rows:",
    f"{final_duplicate_rows:,}",
    flush=True,
)

if final_duplicate_rows > 0:
    raise ValueError(
        "Final output contains duplicate account-month rows."
    )

print(
    "\nFinal columns:",
    flush=True,
)

print(
    df_clean.columns.tolist(),
    flush=True,
)


###############################################################################
# SAVE
###############################################################################

df_clean.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nSaved merged and cleaned data to: {OUTPUT_FILE}",
    flush=True,
)