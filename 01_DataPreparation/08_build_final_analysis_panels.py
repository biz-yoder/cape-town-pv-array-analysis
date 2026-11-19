import os
import pandas as pd
import numpy as np


###############################################################################
# CONFIG
###############################################################################

INPUT_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "7_out_09132026/combined_merged_loadshed_time.parquet"
)

OUTPUT_DIR = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5"
)

PREPAID_FILE = os.path.join(
    OUTPUT_DIR,
    "prepaid_cleaned_vector_noagg.parquet",
)

POSTPAID_FILE = os.path.join(
    OUTPUT_DIR,
    "postpaid_cleaned_vector_noagg.parquet",
)

MAX_CONTRACTS_PER_PV_BUILDING = 5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


###############################################################################
# LOAD DATA
###############################################################################

print("=" * 90)
print("LOADING STEP 7 DATA")
print("=" * 90)

data = pd.read_parquet(
    INPUT_FILE
)

print(
    "Rows loaded:",
    f"{len(data):,}",
)

print(
    "Unique accounts:",
    f"{data['contract_account_hashed'].nunique():,}",
)

print(
    "Household PV eligibility rule: "
    f"maximum contracts per building <= "
    f"{MAX_CONTRACTS_PER_PV_BUILDING}"
)


###############################################################################
# BASIC PREPROCESSING
###############################################################################

print("\n" + "=" * 90)
print("BASIC PREPROCESSING")
print("=" * 90)

data["total_duration_hr"] = (
    data["total_duration_hours"]
)

data["total_peak_hr"] = (
    data["peak_duration_hours"]
)

data["total_standard_hr"] = (
    data["standard_duration_hours"]
)

data["total_off_peak_hr"] = (
    data["off_peak_duration_hours"]
)

data["month_year"] = pd.to_datetime(
    data["month_year"],
    errors="coerce",
)

invalid_months = (
    data["month_year"]
    .isna()
    .sum()
)

print(
    "Rows with invalid month_year:",
    f"{invalid_months:,}",
)

if invalid_months > 0:
    raise ValueError(
        "Invalid month_year values found."
    )

data["month"] = (
    data["month_year"].dt.month
)

data["year"] = (
    data["month_year"].dt.year
)


###############################################################################
# CLEAN SHS LABEL
###############################################################################

print("\n" + "=" * 90)
print("CLEANING SHS LABEL")
print("=" * 90)

data = data.loc[
    ~data["shs_label_edit"].isin(
        [
            "PV_pool",
            "PV_heater",
        ]
    )
].copy()

data["shs_label_edit"] = (
    data["shs_label_edit"]
    .astype("category")
)

if (
    "none"
    not in data[
        "shs_label_edit"
    ].cat.categories
):
    data["shs_label_edit"] = (
        data["shs_label_edit"]
        .cat.add_categories(
            ["none"]
        )
    )

data["shs_label_edit"] = (
    data["shs_label_edit"]
    .fillna("none")
)

data["shs_label_edit"] = (
    data["shs_label_edit"]
    .cat.reorder_categories(
        [
            "none"
        ]
        + [
            x
            for x
            in data[
                "shs_label_edit"
            ].cat.categories
            if x != "none"
        ],
        ordered=True,
    )
)


###############################################################################
# CONSTRUCT CONTRACT-LEVEL PV CAPACITY
###############################################################################

print("\n" + "=" * 90)
print("CONSTRUCTING PV CAPACITY")
print("=" * 90)

data["PV_kW"] = (
    pd.to_numeric(
        data[
            "PV_capacity_W_per_contract"
        ],
        errors="coerce",
    )
    / 1000.0
)


###############################################################################
# DETERMINE MAXIMUM NUMBER OF CONTRACTS PER BUILDING
###############################################################################

print("\n" + "=" * 90)
print("CALCULATING BUILDING-LEVEL CONTRACT COUNTS")
print("=" * 90)

required_building_cols = [
    "building_id",
    "n_contracts_building_month",
]

missing_building_cols = [
    col
    for col
    in required_building_cols
    if col not in data.columns
]

if missing_building_cols:
    raise KeyError(
        "Missing columns required for "
        "PV building cutoff: "
        f"{missing_building_cols}"
    )

data[
    "n_contracts_building_month"
] = pd.to_numeric(
    data[
        "n_contracts_building_month"
    ],
    errors="coerce",
)

invalid_contract_count = (
    data["building_id"].notna()
    & (
        data[
            "n_contracts_building_month"
        ].isna()
        | data[
            "n_contracts_building_month"
        ].le(0)
    )
)

print(
    "Rows with building ID but missing/nonpositive "
    "building-month contract count:",
    f"{invalid_contract_count.sum():,}",
)

if invalid_contract_count.any():
    raise ValueError(
        "Some building-linked observations have "
        "invalid n_contracts_building_month."
    )

building_max_contracts = (
    data.loc[
        data["building_id"].notna(),
        [
            "building_id",
            "n_contracts_building_month",
        ],
    ]
    .groupby(
        "building_id"
    )[
        "n_contracts_building_month"
    ]
    .max()
    .rename(
        "max_contracts_building_panel"
    )
)

data[
    "max_contracts_building_panel"
] = (
    data[
        "building_id"
    ].map(
        building_max_contracts
    )
)

print(
    "Unique buildings:",
    f"{len(building_max_contracts):,}",
)

print(
    "\nMaximum contracts per building:"
)

print(
    building_max_contracts.describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


###############################################################################
# APPLY HOUSEHOLD-LEVEL PV BUILDING CUTOFF
###############################################################################

print("\n" + "=" * 90)
print("APPLYING HOUSEHOLD-LEVEL PV BUILDING CUTOFF")
print("=" * 90)

pv_before_cutoff = (
    data["has_shs"].eq(1)
)

n_pv_contracts_before = data.loc[
    pv_before_cutoff,
    "contract_account_hashed"
].nunique()

n_pv_buildings_before = data.loc[
    pv_before_cutoff,
    "building_id"
].nunique()

print(
    "PV household-months before building cutoff:",
    f"{pv_before_cutoff.sum():,}",
)

print(
    "Unique PV contracts before building cutoff:",
    f"{n_pv_contracts_before:,}",
)

print(
    "Unique PV buildings before building cutoff:",
    f"{n_pv_buildings_before:,}",
)

pv_building_summary_before = (
    data.loc[
        pv_before_cutoff
        & data["building_id"].notna(),
        [
            "building_id",
            "max_contracts_building_panel",
        ],
    ]
    .drop_duplicates(
        "building_id"
    )
)

print(
    "\nMaximum contracts per detected-PV building "
    "before cutoff:"
)

print(
    pv_building_summary_before[
        "max_contracts_building_panel"
    ].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


###############################################################################
# IDENTIFY INVALID HOUSEHOLD-LEVEL PV OBSERVATIONS
###############################################################################

invalid_household_pv = (
    pv_before_cutoff
    & data[
        "max_contracts_building_panel"
    ].gt(
        MAX_CONTRACTS_PER_PV_BUILDING
    )
)

n_pv_contracts_affected = data.loc[
    invalid_household_pv,
    "contract_account_hashed"
].nunique()

n_pv_buildings_excluded = data.loc[
    invalid_household_pv,
    "building_id"
].nunique()

print(
    "\nPV household-months removed by building cutoff:",
    f"{invalid_household_pv.sum():,}",
)

print(
    "Unique PV contracts affected:",
    f"{n_pv_contracts_affected:,}",
)

print(
    "Unique PV buildings excluded:",
    f"{n_pv_buildings_excluded:,}",
)


###############################################################################
# TURN OFF HOUSEHOLD PV FOR INELIGIBLE BUILDINGS
###############################################################################

data.loc[
    invalid_household_pv,
    "has_shs",
] = 0

data.loc[
    invalid_household_pv,
    "shs_label_edit",
] = "none"

data.loc[
    invalid_household_pv,
    "shs_area_m2_edit",
] = np.nan

data.loc[
    invalid_household_pv,
    "PV_kW",
] = 0.0

data.loc[
    invalid_household_pv,
    "PV_capacity_W_per_contract",
] = np.nan

if "Watt" in data.columns:

    data.loc[
        invalid_household_pv,
        "Watt",
    ] = np.nan

if "capacity_source" in data.columns:

    data.loc[
        invalid_household_pv,
        "capacity_source",
    ] = np.nan


###############################################################################
# QA AFTER BUILDING CUTOFF
###############################################################################

pv_after_cutoff = (
    data["has_shs"].eq(1)
)

n_pv_contracts_after = data.loc[
    pv_after_cutoff,
    "contract_account_hashed"
].nunique()

n_pv_buildings_after = data.loc[
    pv_after_cutoff,
    "building_id"
].nunique()

print(
    "\nPV household-months after building cutoff:",
    f"{pv_after_cutoff.sum():,}",
)

print(
    "Unique PV contracts after building cutoff:",
    f"{n_pv_contracts_after:,}",
)

print(
    "Unique PV buildings after building cutoff:",
    f"{n_pv_buildings_after:,}",
)

remaining_invalid_pv = (
    data["has_shs"].eq(1)
    & data[
        "max_contracts_building_panel"
    ].gt(
        MAX_CONTRACTS_PER_PV_BUILDING
    )
)

print(
    "Remaining PV rows in buildings with "
    f">{MAX_CONTRACTS_PER_PV_BUILDING} contracts:",
    f"{remaining_invalid_pv.sum():,}",
)

if remaining_invalid_pv.any():

    raise ValueError(
        "PV observations remain in buildings "
        "exceeding the maximum-contract cutoff."
    )


###############################################################################
# FIRST-PV BACKFILL
###############################################################################

print("\n" + "=" * 90)
print("FIRST-PV BACKFILL")
print("=" * 90)

yearly_pv_counts = (
    data.loc[
        data["has_shs"].eq(1)
    ]
    .groupby(
        "year"
    )[
        "contract_account_hashed"
    ]
    .nunique()
)

print(
    "\nPV households by year BEFORE first-PV backfill:"
)

print(
    yearly_pv_counts
)


###############################################################################
# FIND FIRST ELIGIBLE PV MONTH FOR EACH ACCOUNT
###############################################################################

pv_mask = (
    data["has_shs"].eq(1)
)

first_pv_idx = (
    data.loc[
        pv_mask
    ]
    .groupby(
        "contract_account_hashed"
    )[
        "month_year"
    ]
    .idxmin()
)

first_pv = data.loc[
    first_pv_idx,
    [
        "contract_account_hashed",
        "month_year",
        "shs_label_edit",
        "shs_area_m2_edit",
        "PV_kW",
    ],
].copy()

first_pv = first_pv.rename(
    columns={
        "month_year":
            "first_pv_month",

        "shs_label_edit":
            "first_pv_label",

        "shs_area_m2_edit":
            "first_pv_area",

        "PV_kW":
            "first_pv_kw",
    }
)


###############################################################################
# BACKFILL START DATE
###############################################################################

first_pv[
    "backfill_start"
] = first_pv[
    "first_pv_month"
].apply(
    lambda value: pd.Timestamp(
        year=value.year - 1,
        month=6,
        day=1,
    )
)

data = data.merge(
    first_pv,
    on="contract_account_hashed",
    how="left",
    validate="many_to_one",
)


###############################################################################
# IDENTIFY ELIGIBLE BACKFILL ROWS
###############################################################################

backfill_mask = (
    data[
        "backfill_start"
    ].notna()

    & data[
        "month_year"
    ].ge(
        data[
            "backfill_start"
        ]
    )

    & data[
        "month_year"
    ].lt(
        data[
            "first_pv_month"
        ]
    )
)

print(
    "Contract-month rows in temporal backfill window:",
    f"{backfill_mask.sum():,}",
)

eligible_building_for_backfill = (
    data[
        "max_contracts_building_panel"
    ].le(
        MAX_CONTRACTS_PER_PV_BUILDING
    )
)

backfill_mask = (
    backfill_mask
    & eligible_building_for_backfill
)

print(
    "Contract-month rows in temporal window "
    "AND eligible building:",
    f"{backfill_mask.sum():,}",
)

backfill_missing_shs = (
    backfill_mask
    & ~data["has_shs"].eq(1)
)

print(
    "Contract-month rows actually changed to SHS:",
    f"{backfill_missing_shs.sum():,}",
)


###############################################################################
# APPLY FIRST-PV BACKFILL
###############################################################################

data.loc[
    backfill_missing_shs,
    "has_shs",
] = 1

data.loc[
    backfill_missing_shs,
    "shs_label_edit",
] = "PV_normal"

data.loc[
    backfill_missing_shs,
    "shs_area_m2_edit",
] = data.loc[
    backfill_missing_shs,
    "first_pv_area",
]

data.loc[
    backfill_missing_shs,
    "PV_kW",
] = data.loc[
    backfill_missing_shs,
    "first_pv_kw",
]


###############################################################################
# QA: LATER OBSERVED PV WAS NOT OVERWRITTEN
###############################################################################

post_first_pv = (
    data[
        "first_pv_month"
    ].notna()

    & data[
        "month_year"
    ].ge(
        data[
            "first_pv_month"
        ]
    )
)

print(
    "Rows at/after first PV observation "
    "left untouched by backfill:",
    f"{post_first_pv.sum():,}",
)


###############################################################################
# DROP TEMPORARY BACKFILL VARIABLES
###############################################################################

data = data.drop(
    columns=[
        "first_pv_month",
        "first_pv_label",
        "first_pv_area",
        "first_pv_kw",
        "backfill_start",
    ]
)


###############################################################################
# FINAL PV QA AFTER BACKFILL
###############################################################################

yearly_pv_counts_after = (
    data.loc[
        data["has_shs"].eq(1)
    ]
    .groupby(
        "year"
    )[
        "contract_account_hashed"
    ]
    .nunique()
)

print(
    "\nPV households by year AFTER first-PV backfill:"
)

print(
    yearly_pv_counts_after
)

invalid_after_backfill = (
    data["has_shs"].eq(1)
    & data[
        "max_contracts_building_panel"
    ].gt(
        MAX_CONTRACTS_PER_PV_BUILDING
    )
)

print(
    "\nPV rows in buildings with "
    f">{MAX_CONTRACTS_PER_PV_BUILDING} contracts "
    "AFTER backfill:",
    f"{invalid_after_backfill.sum():,}",
)

if invalid_after_backfill.any():

    raise ValueError(
        "First-PV backfill created PV observations "
        "on an ineligible building."
    )

pv_missing_capacity = (
    data["has_shs"].eq(1)
    & (
        data["PV_kW"].isna()
        | data["PV_kW"].le(0)
    )
)

print(
    "PV rows with missing/nonpositive PV_kW "
    "after cutoff and backfill:",
    f"{pv_missing_capacity.sum():,}",
)

if pv_missing_capacity.any():

    raise ValueError(
        "Some retained PV observations have "
        "missing or nonpositive PV_kW."
    )


###############################################################################
# TARIFF CATEGORIES
###############################################################################

print("\n" + "=" * 90)
print("CREATING TARIFF CATEGORIES")
print("=" * 90)

data.reset_index(
    drop=True,
    inplace=True,
)


def categorize_tariff(name):

    if pd.isna(name):
        return np.nan

    if "Domestic" in name:
        return "Domestic"

    elif "HomeUser" in name:
        return "Home User"

    elif "LifeLine" in name:
        return "LifeLine"

    else:
        return np.nan


data[
    "tariff_category"
] = (
    data["trfname"]
    .apply(
        categorize_tariff
    )
)

data[
    "tariff_category"
] = (
    data
    .groupby(
        "contract_account_hashed"
    )[
        "tariff_category"
    ]
    .transform(
        lambda x:
            x.ffill().bfill()
    )
)


###############################################################################
# REBUILD FINAL SHS VARIABLES
###############################################################################

print("\n" + "=" * 90)
print("REBUILDING FINAL SHS VARIABLES")
print("=" * 90)

data[
    "shs_label_numeric"
] = (
    data[
        "shs_label_edit"
    ]
    .map(
        {
            "PV_normal": 1,
            "none": 0,
        }
    )
    .fillna(0)
    .astype(int)
)

shs_disagreement = (
    data[
        "has_shs"
    ].astype(int)
    != data[
        "shs_label_numeric"
    ].astype(int)
)

print(
    "Rows where has_shs and "
    "shs_label_numeric disagree:",
    f"{shs_disagreement.sum():,}",
)

if shs_disagreement.any():

    raise ValueError(
        "has_shs and shs_label_numeric disagree."
    )


###############################################################################
# FINAL CAPACITY CONSISTENCY
###############################################################################

nonpv_positive_capacity = (
    data[
        "shs_label_numeric"
    ].eq(0)

    & data[
        "PV_kW"
    ].fillna(0)
    .gt(0)
)

print(
    "Non-PV rows with positive PV_kW:",
    f"{nonpv_positive_capacity.sum():,}",
)

if nonpv_positive_capacity.any():

    raise ValueError(
        "Non-PV observations retain positive PV_kW."
    )

data.loc[
    data[
        "shs_label_numeric"
    ].eq(0),
    "PV_kW",
] = 0.0


###############################################################################
# CREATE INTERACTION VARIABLES
###############################################################################

data[
    "interaction_duration_shs_label"
] = (
    data[
        "total_duration_hr"
    ]
    * data[
        "shs_label_numeric"
    ]
)

data[
    "interaction_duration_PV_kW"
] = (
    data[
        "total_duration_hr"
    ]
    * data[
        "PV_kW"
    ]
)


###############################################################################
# FINAL CUTOFF QA
###############################################################################

print("\n" + "=" * 90)
print("FINAL HOUSEHOLD-PV CUTOFF QA")
print("=" * 90)

final_pv = (
    data[
        "shs_label_numeric"
    ].eq(1)
)

n_final_pv_contracts = data.loc[
    final_pv,
    "contract_account_hashed"
].nunique()

n_final_pv_buildings = data.loc[
    final_pv,
    "building_id"
].nunique()

print(
    "Final PV household-months:",
    f"{final_pv.sum():,}",
)

print(
    "Final unique PV contracts:",
    f"{n_final_pv_contracts:,}",
)

print(
    "Final unique PV buildings:",
    f"{n_final_pv_buildings:,}",
)

final_bad_building = (
    final_pv
    & data[
        "max_contracts_building_panel"
    ].gt(
        MAX_CONTRACTS_PER_PV_BUILDING
    )
)

print(
    "Final PV rows in buildings above cutoff:",
    f"{final_bad_building.sum():,}",
)

if final_bad_building.any():

    raise ValueError(
        "Final Step-8 data contain PV rows "
        "in buildings above the cutoff."
    )

print(
    "\nFinal retained-PV building maximum-contract distribution:"
)

print(
    data.loc[
        final_pv,
        [
            "building_id",
            "max_contracts_building_panel",
        ],
    ]
    .drop_duplicates(
        "building_id"
    )[
        "max_contracts_building_panel"
    ]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


###############################################################################
# SPLIT PREPAID / POSTPAID
###############################################################################

print("\n" + "=" * 90)
print("SPLITTING PREPAID AND POSTPAID")
print("=" * 90)


def clean_type(
    df,
    ttype,
    tariffs=None,
):

    df_type = df.loc[
        df["Type"].eq(
            ttype
        )
    ].copy()

    if ttype == "prepaid":
        col = "trfname"
    else:
        col = None

    if (
        tariffs is not None
        and col is not None
        and col in df_type.columns
    ):

        df_type = df_type.loc[
            df_type[
                col
            ].isin(
                tariffs
            )
        ].copy()

    duplicate_rows = (
        df_type.duplicated(
            subset=[
                "contract_account_hashed",
                "month_year",
            ],
            keep=False,
        )
        .sum()
    )

    print(
        f"{ttype} rows involved in "
        "duplicate account-months:",
        f"{duplicate_rows:,}",
    )

    if duplicate_rows > 0:

        raise ValueError(
            f"{ttype} contains duplicate "
            "account-month rows."
        )

    return df_type


prepaid = clean_type(
    data,
    "prepaid",
)

postpaid = clean_type(
    data,
    "postpaid",
)


###############################################################################
# SHS CHECKS
###############################################################################

def shs_check(
    df,
    label="PV_normal",
    type_name=None,
):

    if type_name:

        print(
            f"\n{type_name} SHS Check"
        )

    df_all = (
        df
        .groupby(
            "year"
        )[
            "contract_account_hashed"
        ]
        .nunique()
        .reset_index()
    )

    df_all = df_all.rename(
        columns={
            "contract_account_hashed":
                "unique_contracts"
        }
    )

    print(
        "All contracts per year:"
    )

    print(
        df_all
    )

    df_shs = (
        df.loc[
            df[
                "shs_label_edit"
            ].eq(
                label
            )
        ]
        .groupby(
            "year"
        )[
            "contract_account_hashed"
        ]
        .nunique()
        .reset_index()
    )

    df_shs = df_shs.rename(
        columns={
            "contract_account_hashed":
                "unique_contracts"
        }
    )

    print(
        "SHS contracts per year:"
    )

    print(
        df_shs
    )


shs_check(
    data,
    type_name="All",
)

shs_check(
    prepaid,
    type_name="Prepaid",
)

shs_check(
    postpaid,
    type_name="Postpaid",
)


###############################################################################
# FINAL PREPAID / POSTPAID PV QA
###############################################################################

print("\n" + "=" * 90)
print("FINAL PREPAID / POSTPAID PV QA")
print("=" * 90)

for name, df in [
    ("Prepaid", prepaid),
    ("Postpaid", postpaid),
]:

    pv_mask_final = (
        df[
            "shs_label_numeric"
        ].eq(1)
    )

    bad = (
        pv_mask_final
        & df[
            "max_contracts_building_panel"
        ].gt(
            MAX_CONTRACTS_PER_PV_BUILDING
        )
    )

    n_unique_pv_accounts = df.loc[
        pv_mask_final,
        "contract_account_hashed"
    ].nunique()

    print(
        f"\n{name}:"
    )

    print(
        "  Rows:",
        f"{len(df):,}",
    )

    print(
        "  Unique accounts:",
        f"{df['contract_account_hashed'].nunique():,}",
    )

    print(
        "  PV household-months:",
        f"{pv_mask_final.sum():,}",
    )

    print(
        "  Unique PV accounts:",
        f"{n_unique_pv_accounts:,}",
    )

    print(
        "  PV rows above building cutoff:",
        f"{bad.sum():,}",
    )

    if bad.any():

        raise ValueError(
            f"{name} output contains PV rows "
            "above the building cutoff."
        )


###############################################################################
# SAVE OUTPUTS
###############################################################################

print("\n" + "=" * 90)
print("SAVING STEP 8 OUTPUTS")
print("=" * 90)

prepaid.to_parquet(
    PREPAID_FILE,
    index=False,
)

postpaid.to_parquet(
    POSTPAID_FILE,
    index=False,
)

print(
    "Saved prepaid:",
    PREPAID_FILE,
)

print(
    "Saved postpaid:",
    POSTPAID_FILE,
)

print(
    "\nHousehold-level PV building rule:",
    f"max_contracts_building_panel <= "
    f"{MAX_CONTRACTS_PER_PV_BUILDING}",
)

print(
    "\nProcessing complete!"
)