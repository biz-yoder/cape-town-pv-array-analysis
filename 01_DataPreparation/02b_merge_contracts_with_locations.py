
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONTRACT_LOCATION_MERGE, HISTORICAL_CONTRACT_LOCATIONS, POSTPAID_MONTHLY, PREPAID_MONTHLY
#Match contracts with locations data

import pandas as pd
from datetime import datetime
import os
import polars as pl

# Paths
locations_path = str(HISTORICAL_CONTRACT_LOCATIONS)
prepaid_path = str(PREPAID_MONTHLY)
postpaid_path = str(POSTPAID_MONTHLY)
output_path = str(CONTRACT_LOCATION_MERGE)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

#######################################
# Load data

locations_df = pd.read_parquet(locations_path)
prepaid_df = pd.read_parquet(prepaid_path)
postpaid_df = pd.read_parquet(postpaid_path)

#######################################
# Baseline reference counts
baseline_rows = len(prepaid_df) + len(postpaid_df)
baseline_contracts = pd.concat([
    prepaid_df["contract_account_hashed"], 
    postpaid_df["contract_hashed"]
    # postpaid_df["account_id"]
]).nunique()

print(f"Baseline counts:")
print(f"  Total rows: {baseline_rows:,}")
print(f"  Unique contracts: {baseline_contracts:,}")
print("="*80)

#######################################
# Prepare columns

prepaid_df['Type'] = 'prepaid'
postpaid_df['Type'] = 'postpaid'

# postpaid_df = postpaid_df.rename(columns={"totalunits": "kwh"})

print("\nAfter column prep:")
print(f"  prepaid_df unique contract_IDs: {prepaid_df['contract_account_hashed'].nunique():,} "
      f"({prepaid_df['contract_account_hashed'].nunique()/baseline_contracts*100:.1f}% of baseline)")
print(f"  postpaid_df unique contract_IDs: {postpaid_df['contract_hashed'].nunique():,} "
      f"({postpaid_df['contract_hashed'].nunique()/baseline_contracts*100:.1f}% of baseline)")
print("="*80)

#######################################
# Combine data

required_prepaid_cols = ['contract_account_hashed', 'contract_ID', 'Type', 'month_year', 'trfname', 'kwh']
required_postpaid_cols = ['contract_hashed', 'contract_ID', 'Type', 'month_year', 'kwh'] #removed rate_category

before_concat = len(prepaid_df) + len(postpaid_df)
postpaid_df["contract_ID"] = postpaid_df["contract_hashed"]
prepaid_df["contract_ID"] = prepaid_df["contract_account_hashed"]
# postpaid_df["contract_ID"] = postpaid_df["account_id"]

df_combined = pd.concat([
    prepaid_df[required_prepaid_cols],
    postpaid_df[required_postpaid_cols]
], ignore_index=True)

print(f"\nAfter concatenation: {len(df_combined):,} rows "
      f"({len(df_combined)/baseline_rows*100:.1f}% of baseline rows), "
      f"unique contracts: {df_combined['contract_ID'].nunique():,} "
      f"({df_combined['contract_ID'].nunique()/baseline_contracts*100:.1f}% of baseline contracts)")
print("="*80)

input_hhmonth_dups = df_combined.loc[
    df_combined.duplicated(
        subset=["contract_ID", "month_year"],
        keep=False
    )
].copy()

print(
    "\nInput household-month rows involved in duplicates:",
    len(input_hhmonth_dups)
)

print(
    "Distinct duplicated input household-months:",
    input_hhmonth_dups[
        ["contract_ID", "month_year"]
    ].drop_duplicates().shape[0]
)

if len(input_hhmonth_dups) > 0:
    input_variation = (
        input_hhmonth_dups
        .groupby(["contract_ID", "month_year"])
        .nunique(dropna=False)
    )

    input_columns_that_differ = (
        input_variation.gt(1)
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "\nColumns differing within input household-month duplicates:"
    )
    print(
        input_columns_that_differ[
            input_columns_that_differ > 0
        ].head(20)
    )

#######################################
# Fix dates

df_combined["month_year"] = pd.to_datetime(df_combined["month_year"], format="%Y-%m", errors="coerce")
valid_dates = df_combined["month_year"].notna().sum()
invalid_dates = df_combined["month_year"].isna().sum()
print(f"Date conversion: {valid_dates:,} valid, {invalid_dates:,} invalid "
      f"({valid_dates/baseline_rows*100:.1f}% of baseline rows)")
print("="*80)

#######################################
# Clean locations data

before_loc = len(locations_df)

locations_df["move_in_timestamp"] = pd.to_datetime(locations_df["move_in_timestamp"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
locations_df["move_out_timestamp"] = pd.to_datetime(locations_df["move_out_timestamp"], errors="coerce")
locations_df.loc[locations_df["move_out_timestamp"].isna(), "move_out_timestamp"] = pd.Timestamp(datetime.today().date())
locations_df.loc[locations_df["move_out_timestamp"].dt.year == 9999, "move_out_timestamp"] = pd.Timestamp(datetime.today().date())

mask_bad = locations_df["move_out_timestamp"] < locations_df["move_in_timestamp"]
if mask_bad.any():
    print(f"⚠️ Fixing {mask_bad.sum():,} rows where move_out < move_in")
    locations_df.loc[mask_bad, "move_out_timestamp"] = pd.Timestamp(datetime.today().date())

# print(f"Cleaned locations_df: {len(locations_df):,} rows (was {before_loc:,}), "
#       f"unique contracts: {locations_df['contract_account_hashed'].nunique():,} "
#       f"({locations_df['contract_account_hashed'].nunique()/baseline_contracts*100:.1f}% of baseline)")
# print("="*80)

#######################################
# Merge contract data with locations

df_combined_pl = pl.from_pandas(df_combined)
locations_pl = pl.from_pandas(locations_df)

#######################################
# Prepare col names for merge

locations_pl_account = (
    locations_pl.with_columns([
        pl.col("contract_account_hashed").alias("contract_account_hashed_loc")
    ])
)

locations_pl_contract = (
    locations_pl.with_columns([
        pl.col("contract_account_hashed").alias("contract_account_hashed_loc"),
        pl.col("contract_hashed").alias("contract_hashed_loc")
    ])
)

#######################################
# Merge: New data (with contract_account_hashed)

df1 = (
    df_combined_pl.join(
        locations_pl_account,
        left_on="contract_account_hashed",
        right_on="contract_account_hashed_loc",
        how="inner"
    )
    .filter(
        (pl.col("month_year") >= pl.col("move_in_timestamp")) &
        (pl.col("month_year") <= pl.col("move_out_timestamp"))
    )
    .with_columns(pl.lit("account_match").alias("match_type"))
)
print(f"After account join: {df1.height:,} rows")

matched_ids = df1.select("contract_ID").unique()

#######################################
# Merge: Postpaid data (with contract_hashed)

# Get matched contract_IDs as a Series
matched_ids = df1.select("contract_ID").unique()["contract_ID"]

df2 = (
    df_combined_pl
    .filter(~pl.col("contract_ID").is_in(matched_ids))
    .join(
        locations_pl_contract,
        left_on="contract_hashed",
        right_on="contract_hashed_loc",
        how="inner"
    )
    .filter(
        (pl.col("month_year") >= pl.col("move_in_timestamp")) &
        (pl.col("month_year") <= pl.col("move_out_timestamp"))
    )
    .with_columns(pl.lit("contract_match").alias("match_type"))
)

print(f"After contract join: {df2.height:,} rows")

#######################################
# Combine merges

df_merged = pl.concat([df1, df2], how="diagonal_relaxed")

#######################################
# Combine the two 'contract_account_hashed' sources into one

df_merged = df_merged.with_columns(
    pl.coalesce([
        pl.col("contract_account_hashed"),          # original
        pl.col("contract_account_hashed_loc"),      # from merge 1
    ]).alias("contract_account_hashed")
)

cols_to_drop = [c for c in ["contract_account_hashed_loc", "contract_hashed_loc", "contract_account_hashed_device"] if c in df_merged.columns]
df_merged = df_merged.drop(cols_to_drop)

#######################################
# Convert to pandas

df_merged = df_merged.to_pandas()

df_merged["contract_ID"] = df_merged["contract_account_hashed"].combine_first(df_merged["contract_hashed"])

# -------------------------------------------------
# Resolve multiple valid location matches
# -------------------------------------------------

df_merged["match_priority"] = (
    df_merged["match_type"]
    .map({
        "account_match": 0,
        "contract_match": 1
    })
    .fillna(2)
)

df_merged["has_wkt"] = df_merged["wkt"].notna().astype(int)

if "active" in df_merged.columns:
    df_merged["active_priority"] = (
        df_merged["active"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "active"])
        .astype(int)
    )
else:
    df_merged["active_priority"] = 0

print(
    "\nActive values:"
)
print(
    df_merged["active"]
    .astype(str)
    .value_counts(dropna=False)
    .head(20)
)

df_merged = df_merged.sort_values(
    [
        "contract_ID",
        "month_year",

        # Prefer account-level matches
        "match_priority",

        # Prefer records with geometry
        "has_wkt",

        # Prefer active records
        "active_priority",

        # Among valid histories, use the most recent move-in
        "move_in_timestamp",

        # Final tie-breaker
        "move_out_timestamp",
    ],
    ascending=[
        True,
        True,
        True,
        False,
        False,
        False,
        True,
    ],
    na_position="last"
)

df_merged = df_merged.drop_duplicates(
    subset=["contract_ID", "month_year"],
    keep="first"
)

df_merged = df_merged.drop(
    columns=[
        "match_priority",
        "has_wkt",
        "active_priority",
    ],
    errors="ignore"
)

#######################################
# Fill missing location data 

loc_cols = ["ward2021", "move_in_timestamp", "move_out_timestamp", "wkt"]
loc_cols = [c for c in loc_cols if c in df_merged.columns]
df_merged = df_merged.sort_values(["contract_ID", "month_year"])
df_merged[loc_cols] = df_merged.groupby("contract_ID")[loc_cols].ffill()
df_merged[loc_cols] = df_merged.groupby("contract_ID")[loc_cols].bfill()

print(f"Missing location values after fill: {df_merged[loc_cols].isna().sum().to_dict()} "
      f"({df_merged['wkt'].notna().mean()*100:.1f}% rows have location)")

#######################################
# Fix duplicates

print("\nResolving duplicates...")

dup_before = df_merged.duplicated(keep=False).sum()
print(f"Identical duplicate rows before drop: {dup_before:,}")

df_merged = df_merged.drop_duplicates()

hhmonth_dup_after = df_merged.duplicated(
    subset=["contract_ID", "month_year"],
    keep=False
).sum()

hhmonth_combo_after = (
    df_merged.loc[
        df_merged.duplicated(
            subset=["contract_ID", "month_year"],
            keep=False
        ),
        ["contract_ID", "month_year"]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Household-month rows still duplicated after removing "
    f"identical rows: {hhmonth_dup_after:,}"
)

print(
    f"Distinct household-month combinations still duplicated: "
    f"{hhmonth_combo_after:,}"
)

dup_after = df_merged.duplicated(keep=False).sum()

print(f"Identical duplicate rows after drop: {dup_after:,}")
print(f"Remaining rows: {len(df_merged):,} ({len(df_merged)/baseline_rows*100:.1f}% of baseline)")
print(f"Remaining unique contracts: {df_merged['contract_ID'].nunique():,} "
      f"({df_merged['contract_ID'].nunique()/baseline_contracts*100:.1f}% of baseline)")

if hhmonth_dup_after > 0:

    remaining_hhmonth_dups = df_merged.loc[
        df_merged.duplicated(
            subset=["contract_ID","month_year"],
            keep=False
        )
    ].copy()

    variation = (
        remaining_hhmonth_dups
        .groupby(["contract_ID","month_year"])
        .nunique(dropna=False)
    )

    columns_that_differ = (
        variation.gt(1)
        .sum()
        .sort_values(ascending=False)
    )

    print("\nColumns differing within remaining household-month duplicates:")
    print(columns_that_differ[columns_that_differ > 0].head(30))

else:
    print("\nNo household-month duplicates remain.")
#######################################
# Summarize location coverage

contracts_with_wkt = df_merged.loc[df_merged["wkt"].notna(), ["contract_ID", "Type"]].drop_duplicates()
contracts_total = df_merged[["contract_ID", "Type"]].drop_duplicates()

summary = (
    contracts_total.groupby("Type")
    .agg(total_contracts=("contract_ID", "count"))
    .reset_index()
    .merge(
        contracts_with_wkt.groupby("Type").agg(with_location=("contract_ID", "count")).reset_index(),
        on="Type",
        how="left"
    )
    .fillna(0)
)
summary["percent_with_location"] = (summary["with_location"] / summary["total_contracts"]) * 100
summary["percent_of_baseline_contracts"] = (summary["total_contracts"] / baseline_contracts) * 100
print("\nUnique contracts with location info by Type:")
print(summary.to_string(index=False))
print("="*80)

#######################################
# Save

df_merged = df_merged[df_merged['month_year'] <= '2023-12']
df_merged.to_parquet(output_path, index=False)
print(f"\n✅ Saved final dataset to: {output_path}")
print(f"Final rows: {len(df_merged):,} ({len(df_merged)/baseline_rows*100:.1f}% of baseline)")
print(f"Final unique contracts: {df_merged['contract_ID'].nunique():,} "
      f"({df_merged['contract_ID'].nunique()/baseline_contracts*100:.1f}% of baseline)")
print("="*80)
