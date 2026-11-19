# Add monthly load-shedding duration measures to each contract-month observation.

from __future__ import annotations

import os
import re
import time

import pandas as pd


###############################################################################
# PATHS
###############################################################################

LOADSHED_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "Loadshedding_schedule.csv"
)

PANEL_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "6_out_09132026/merged_with_blocks_combined.parquet"
)

OUTPUT_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "7_out_09132026/combined_merged_loadshed_time.parquet"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

###############################################################################
# HELPERS
###############################################################################

def section(title: str) -> None:
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


def extract_area_number(row):
    """
    Extract the integer following 'Area' from:
      1. Area column, then
      2. Stage column.
    """
    area_pattern = re.compile(r"area\s*([0-9]+)", flags=re.IGNORECASE)

    area_value = str(row.get("Area", ""))
    stage_value = str(row.get("Stage", ""))

    area_match = area_pattern.search(area_value)
    if area_match:
        return int(area_match.group(1))

    stage_match = area_pattern.search(stage_value)
    if stage_match:
        return int(stage_match.group(1))

    return pd.NA


###############################################################################
# LOAD AND CLEAN LOAD-SHEDDING SCHEDULE
###############################################################################

section("LOADING LOAD-SHEDDING SCHEDULE")

shed_df = pd.read_csv(LOADSHED_FILE)
shed_df.columns = shed_df.columns.str.strip()

if "Duration min" in shed_df.columns:
    shed_df = shed_df.rename(columns={"Duration min": "Duration_min"})

required_schedule_columns = [
    "Date",
    "Time",
    "Stage",
    "Duration_min",
]

missing_schedule_columns = [
    column
    for column in required_schedule_columns
    if column not in shed_df.columns
]

if missing_schedule_columns:
    raise KeyError(
        "Load-shedding schedule is missing columns: "
        + str(missing_schedule_columns)
    )

raw_schedule_rows = len(shed_df)
print("Raw schedule rows:", f"{raw_schedule_rows:,}", flush=True)

shed_df["Date"] = pd.to_datetime(shed_df["Date"], errors="coerce")

invalid_dates = int(shed_df["Date"].isna().sum())
print("Rows with invalid schedule Date:", f"{invalid_dates:,}", flush=True)

# Drop any remaining rows with unknown duration
n_dropped = shed_df["Duration_min"].isna().sum()

print(f"Dropping {n_dropped} schedule rows with unresolved missing durations.")

shed_df = shed_df.loc[shed_df["Duration_min"].notna()].copy()

if invalid_dates > 0:
    print(
        shed_df.loc[shed_df["Date"].isna()].head(20).to_string(),
        flush=True,
    )
    raise ValueError("Load-shedding schedule contains invalid dates.")

shed_df["Duration_min"] = pd.to_numeric(
    shed_df["Duration_min"],
    errors="coerce",
)

invalid_duration = int(shed_df["Duration_min"].isna().sum())
print("Rows with invalid Duration_min:", f"{invalid_duration:,}", flush=True)

if invalid_duration > 0:
    raise ValueError("Load-shedding schedule contains invalid durations.")

shed_df["month_year"] = (
    shed_df["Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

###############################################################################
# SPECIAL 2018-2019 DEDUPLICATION
###############################################################################

section("CHECKING 2018-2019 SCHEDULE DUPLICATES")

mask_2018_2019 = shed_df["Date"].dt.year.isin([2018, 2019])

df_2018_2019 = shed_df.loc[mask_2018_2019].copy()
df_other = shed_df.loc[~mask_2018_2019].copy()

rows_before_dedup = len(df_2018_2019)

duplicate_early_rows = df_2018_2019.duplicated(
    subset=[
        "Date",
        "Time",
        "month_year",
        "Stage",
    ],
    keep=False,
)

n_duplicate_early_rows = int(duplicate_early_rows.sum())

print("2018-2019 rows:", f"{rows_before_dedup:,}", flush=True)
print(
    "2018-2019 rows involved in duplicate Date-Time-Stage keys:",
    f"{n_duplicate_early_rows:,}",
    flush=True,
)

# Preserve Area during the special early-year deduplication when it exists,
# because Area may be needed to recover the load-shedding block number.
early_group_columns = [
    "Date",
    "Time",
    "month_year",
    "Stage",
]

if "Area" in df_2018_2019.columns:
    early_group_columns.append("Area")

df_2018_2019 = (
    df_2018_2019
    .groupby(
        early_group_columns,
        as_index=False,
        dropna=False,
    )["Duration_min"]
    .max()
)

rows_after_dedup = len(df_2018_2019)
rows_removed_dedup = rows_before_dedup - rows_after_dedup

print(
    "2018-2019 rows after deduplication:",
    f"{rows_after_dedup:,}",
    flush=True,
)
print(
    "2018-2019 rows removed:",
    f"{rows_removed_dedup:,}",
    flush=True,
)

shed_df = pd.concat(
    [
        df_2018_2019,
        df_other,
    ],
    ignore_index=True,
)


###############################################################################
# EXTRACT LOAD-SHEDDING AREA / BLOCK
###############################################################################

section("EXTRACTING LOAD-SHEDDING AREA NUMBERS")

shed_df["Area_number"] = shed_df.apply(
    extract_area_number,
    axis=1,
)

rows_without_area = int(shed_df["Area_number"].isna().sum())

print(
    "Schedule rows with no extracted area:",
    f"{rows_without_area:,}",
    flush=True,
)

if rows_without_area > 0:
    print("\nExamples with no extracted area:", flush=True)

    possible_columns = [
        column
        for column in [
            "Date",
            "Time",
            "Stage",
            "Area",
        ]
        if column in shed_df.columns
    ]

    print(
        shed_df.loc[
            shed_df["Area_number"].isna(),
            possible_columns,
        ]
        .head(30)
        .to_string(index=False),
        flush=True,
    )

shed_df = shed_df.loc[shed_df["Area_number"].notna()].copy()

shed_df["Area_number"] = (
    pd.to_numeric(
        shed_df["Area_number"],
        errors="raise",
    )
    .astype("int16")
)

print("\nExtracted Area_number distribution:", flush=True)
print(
    shed_df["Area_number"].value_counts().sort_index(),
    flush=True,
)


###############################################################################
# CREATE EVENT DATETIME
###############################################################################

section("CREATING LOAD-SHEDDING EVENT DATETIME")

shed_df["datetime"] = pd.to_datetime(
    (
        shed_df["Date"].dt.strftime("%Y-%m-%d")
        + " "
        + shed_df["Time"].astype(str)
    ),
    errors="coerce",
)

invalid_event_datetime = int(shed_df["datetime"].isna().sum())

print(
    "Rows with invalid event datetime:",
    f"{invalid_event_datetime:,}",
    flush=True,
)

if invalid_event_datetime > 0:
    raise ValueError(
        "Some load-shedding event datetimes could not be parsed."
    )

shed_df["weekday"] = shed_df["datetime"].dt.weekday
shed_df["time_in_minutes"] = (
    shed_df["datetime"].dt.hour * 60
    + shed_df["datetime"].dt.minute
)


###############################################################################
# TIME-OF-USE CLASSIFICATION
###############################################################################

section("CLASSIFYING LOAD-SHEDDING BY TIME OF USE")

# Applicable to the 2018-2023 analysis period:
# Peak: weekdays 07:00-10:00 and 18:00-20:00
# Standard: weekdays 06:00-07:00, 10:00-18:00, and 20:00-22:00
# Off-peak: all other times.

weekday_mask = shed_df["weekday"] < 5

peak_mask = (
    weekday_mask
    & (
        shed_df["time_in_minutes"].between(420, 599)
        | shed_df["time_in_minutes"].between(1080, 1199)
    )
)

standard_mask = (
    weekday_mask
    & (
        shed_df["time_in_minutes"].between(360, 419)
        | shed_df["time_in_minutes"].between(600, 1079)
        | shed_df["time_in_minutes"].between(1200, 1319)
    )
)

off_peak_mask = ~(peak_mask | standard_mask)

shed_df["Duration_min_peak"] = 0.0
shed_df["Duration_min_standard"] = 0.0
shed_df["Duration_min_off_peak"] = 0.0

shed_df.loc[
    peak_mask,
    "Duration_min_peak",
] = shed_df.loc[
    peak_mask,
    "Duration_min",
]

shed_df.loc[
    standard_mask,
    "Duration_min_standard",
] = shed_df.loc[
    standard_mask,
    "Duration_min",
]

shed_df.loc[
    off_peak_mask,
    "Duration_min_off_peak",
] = shed_df.loc[
    off_peak_mask,
    "Duration_min",
]

category_sum = (
    shed_df["Duration_min_peak"]
    + shed_df["Duration_min_standard"]
    + shed_df["Duration_min_off_peak"]
)

duration_difference = category_sum - shed_df["Duration_min"]

max_event_difference = float(duration_difference.abs().max())

print(
    "Maximum event-level TOU duration discrepancy:",
    f"{max_event_difference:.10f}",
    flush=True,
)

if max_event_difference > 1e-8:
    raise ValueError(
        "TOU classification does not sum to total event duration."
    )


###############################################################################
# SUMMARIZE BY BLOCK-MONTH
###############################################################################

section("SUMMARIZING LOAD SHEDDING BY BLOCK-MONTH")

shed_summary = (
    shed_df
    .groupby(
        [
            "Area_number",
            "month_year",
        ],
        as_index=False,
    )
    .agg(
        total_duration_min=("Duration_min", "sum"),
        total_peak_min=("Duration_min_peak", "sum"),
        total_standard_min=("Duration_min_standard", "sum"),
        total_off_peak_min=("Duration_min_off_peak", "sum"),
        n_events=("Duration_min", "size"),
    )
)

summary_duplicates = int(
    shed_summary.duplicated(
        subset=[
            "Area_number",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Duplicate block-month rows in schedule summary:",
    f"{summary_duplicates:,}",
    flush=True,
)

if summary_duplicates > 0:
    raise ValueError(
        "Load-shedding summary is not unique by block and month."
    )

shed_summary["tou_sum_min"] = (
    shed_summary["total_peak_min"]
    + shed_summary["total_standard_min"]
    + shed_summary["total_off_peak_min"]
)

summary_difference = (
    shed_summary["tou_sum_min"]
    - shed_summary["total_duration_min"]
)

max_summary_difference = float(summary_difference.abs().max())

print(
    "Maximum block-month TOU discrepancy:",
    f"{max_summary_difference:.10f}",
    flush=True,
)

if max_summary_difference > 1e-8:
    raise ValueError(
        "Monthly TOU totals do not sum to total load-shedding duration."
    )

shed_summary = shed_summary.drop(columns=["tou_sum_min"])

n_block_month_rows = len(shed_summary)

print(
    "Load-shedding block-month rows:",
    f"{n_block_month_rows:,}",
    flush=True,
)

print("\nLoad-shedding duration summary:", flush=True)
print(
    shed_summary[
        [
            "total_duration_min",
            "total_peak_min",
            "total_standard_min",
            "total_off_peak_min",
        ]
    ].describe(),
    flush=True,
)


###############################################################################
# LOAD CONTRACT-MONTH PANEL
###############################################################################

section("LOADING CONTRACT-MONTH PANEL")

start_time = time.time()
df = pd.read_parquet(PANEL_FILE)
panel_load_seconds = time.time() - start_time

print(
    "Panel load time:",
    f"{panel_load_seconds:.1f} seconds",
    flush=True,
)

required_panel_columns = [
    "contract_account_hashed",
    "month_year",
    "BlockID",
]

missing_panel_columns = [
    column
    for column in required_panel_columns
    if column not in df.columns
]

if missing_panel_columns:
    raise KeyError(
        "Panel is missing required columns: "
        + str(missing_panel_columns)
    )

rows_before_merge = len(df)
unique_accounts_before = df["contract_account_hashed"].nunique()

print("Panel rows:", f"{rows_before_merge:,}", flush=True)
print("Unique accounts:", f"{unique_accounts_before:,}", flush=True)

input_duplicate_rows = int(
    df.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Input duplicate account-month rows:",
    f"{input_duplicate_rows:,}",
    flush=True,
)

if input_duplicate_rows > 0:
    raise ValueError("Input panel is not unique by account-month.")


###############################################################################
# PREPARE PANEL MERGE KEYS
###############################################################################

section("PREPARING PANEL MERGE KEYS")

df["month_year"] = (
    pd.to_datetime(
        df["month_year"],
        errors="coerce",
    )
    .dt.to_period("M")
    .dt.to_timestamp()
)

invalid_panel_month = int(df["month_year"].isna().sum())

print(
    "Panel rows with invalid month_year:",
    f"{invalid_panel_month:,}",
    flush=True,
)

if invalid_panel_month > 0:
    raise ValueError("Panel has invalid month_year values.")

df["Area_number"] = (
    pd.to_numeric(
        df["BlockID"],
        errors="coerce",
    )
    .astype("Int16")
)

panel_block_values = sorted(
    df["Area_number"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

schedule_area_values = sorted(
    shed_summary["Area_number"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

print("Unique panel block IDs:", panel_block_values, flush=True)
print("Unique schedule area IDs:", schedule_area_values, flush=True)

panel_blocks = set(panel_block_values)
schedule_blocks = set(schedule_area_values)

panel_blocks_absent_from_schedule = sorted(
    panel_blocks - schedule_blocks
)

schedule_areas_absent_from_panel = sorted(
    schedule_blocks - panel_blocks
)

print(
    "Panel blocks absent from schedule:",
    panel_blocks_absent_from_schedule,
    flush=True,
)
print(
    "Schedule areas absent from panel:",
    schedule_areas_absent_from_panel,
    flush=True,
)


###############################################################################
# MERGE MONTHLY LOAD SHEDDING
###############################################################################

section("MERGING LOAD-SHEDDING DURATIONS")

merged = df.merge(
    shed_summary,
    on=[
        "Area_number",
        "month_year",
    ],
    how="left",
    validate="many_to_one",
)

rows_after_merge = len(merged)

print(
    "Rows before load-shedding merge:",
    f"{rows_before_merge:,}",
    flush=True,
)
print(
    "Rows after load-shedding merge:",
    f"{rows_after_merge:,}",
    flush=True,
)

if rows_after_merge != rows_before_merge:
    raise ValueError(
        "Load-shedding merge changed the number of panel rows."
    )


###############################################################################
# DISTINGUISH ZERO LOAD SHEDDING FROM MISSING BLOCK
###############################################################################

section("HANDLING MONTHS WITHOUT SCHEDULED LOAD SHEDDING")

duration_columns = [
    "total_duration_min",
    "total_peak_min",
    "total_standard_min",
    "total_off_peak_min",
    "n_events",
]

valid_block = merged["Area_number"].notna()
missing_schedule_match = merged["total_duration_min"].isna()
zero_loadshedding = valid_block & missing_schedule_match

n_zero_loadshedding_rows = int(zero_loadshedding.sum())

print(
    "Rows with valid block but no block-month schedule row:",
    f"{n_zero_loadshedding_rows:,}",
    flush=True,
)

print(
    "\nMonths represented in schedule:",
    sorted(shed_summary["month_year"].unique())[:5],
    "...",
    sorted(shed_summary["month_year"].unique())[-5:],
)

print(
    "Number of block-months in schedule:",
    len(shed_summary),
)

panel_months = set(df["month_year"].unique())
schedule_months = set(shed_summary["month_year"].unique())

print(
    "Panel months with no schedule at all:",
    sorted(panel_months - schedule_months)
)

merged.loc[
    zero_loadshedding,
    duration_columns,
] = 0

missing_block_rows = merged["Area_number"].isna()
n_missing_block_rows = int(missing_block_rows.sum())

print(
    "Rows with no assigned load-shedding block:",
    f"{n_missing_block_rows:,}",
    flush=True,
)

print("\nMissing values after fill:")
print(
    merged[
        [
            "total_duration_min",
            "total_peak_min",
            "total_standard_min",
            "total_off_peak_min",
        ]
    ].isna().sum()
)

###############################################################################
# CREATE HOURS AND STAGE-1 EQUIVALENTS
###############################################################################

section("CREATING FINAL LOAD-SHEDDING VARIABLES")

merged["total_duration_hours"] = merged["total_duration_min"] / 60
merged["peak_duration_hours"] = merged["total_peak_min"] / 60
merged["standard_duration_hours"] = merged["total_standard_min"] / 60
merged["off_peak_duration_hours"] = merged["total_off_peak_min"] / 60

# One Stage-1-equivalent event = 2.5 hours.
merged["total_duration_stage1"] = merged["total_duration_hours"] / 2.5
merged["peak_duration_stage1"] = merged["peak_duration_hours"] / 2.5
merged["standard_duration_stage1"] = merged["standard_duration_hours"] / 2.5
merged["off_peak_duration_stage1"] = merged["off_peak_duration_hours"] / 2.5


###############################################################################
# FINAL QA
###############################################################################

section("FINAL QA")

final_duplicate_rows = int(
    merged.duplicated(
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
        "Final dataset contains duplicate account-month rows."
    )

unique_accounts_after = merged["contract_account_hashed"].nunique()

print(
    "Unique accounts before merge:",
    f"{unique_accounts_before:,}",
    flush=True,
)
print(
    "Unique accounts after merge:",
    f"{unique_accounts_after:,}",
    flush=True,
)

if unique_accounts_after != unique_accounts_before:
    raise ValueError(
        "Load-shedding merge changed the number of unique accounts."
    )

duration_check = (
    merged["total_peak_min"].fillna(0)
    + merged["total_standard_min"].fillna(0)
    + merged["total_off_peak_min"].fillna(0)
    - merged["total_duration_min"].fillna(0)
)

max_final_tou_difference = float(duration_check.abs().max())

print(
    "Maximum final TOU-total discrepancy:",
    f"{max_final_tou_difference:.10f}",
    flush=True,
)

if max_final_tou_difference > 1e-8:
    raise ValueError(
        "Final TOU durations do not sum to total duration."
    )

print("\nMonthly total load-shedding hours:", flush=True)
print(
    merged["total_duration_hours"].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    ),
    flush=True,
)

n_missing_duration_rows = int(
    merged["total_duration_hours"].isna().sum()
)

n_accounts_missing_duration = (
    merged.loc[
        merged["total_duration_hours"].isna(),
        "contract_account_hashed",
    ]
    .nunique()
)

print(
    "\nRows with missing total_duration_hours:",
    f"{n_missing_duration_rows:,}",
    flush=True,
)
print(
    "Unique accounts with missing total_duration_hours:",
    f"{n_accounts_missing_duration:,}",
    flush=True,
)

# Missing duration should only occur when the household has no assigned block.
unexpected_missing_duration = (
    merged["total_duration_hours"].isna()
    & merged["Area_number"].notna()
)

n_unexpected_missing_duration = int(
    unexpected_missing_duration.sum()
)

print(
    "Rows with a valid block but missing final duration:",
    f"{n_unexpected_missing_duration:,}",
    flush=True,
)

if n_unexpected_missing_duration > 0:
    raise ValueError(
        "Some valid-block rows still have missing load-shedding duration."
    )

# Verify load-shedding treatment is constant within block-month.
block_month_duration_counts = (
    merged.loc[
        merged["Area_number"].notna()
    ]
    .groupby(
        [
            "Area_number",
            "month_year",
        ]
    )["total_duration_hours"]
    .nunique(dropna=False)
)

max_duration_values_per_block_month = int(
    block_month_duration_counts.max()
)

print(
    "Maximum number of total-duration values within a block-month:",
    f"{max_duration_values_per_block_month:,}",
    flush=True,
)

if max_duration_values_per_block_month > 1:
    raise ValueError(
        "Load-shedding duration varies within at least one block-month."
    )


###############################################################################
# SAVE
###############################################################################

section("SAVING OUTPUT")

merged.to_parquet(
    OUTPUT_FILE,
    engine="pyarrow",
    index=False,
)

saved_rows = len(merged)
saved_columns = len(merged.columns)

print("Saved output:", OUTPUT_FILE, flush=True)
print("Saved rows:", f"{saved_rows:,}", flush=True)
print("Saved columns:", f"{saved_columns:,}", flush=True)