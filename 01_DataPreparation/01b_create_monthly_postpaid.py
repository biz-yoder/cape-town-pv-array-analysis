
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import MONTHLY_DIR, RAW_POSTPAID_PANEL
# Imports all old prepaid transactions (before April 2021) and postpaid data.
# Transforms all transactions into monthly panel data

import polars as pl
import os
from time import time

# Paths
parquet_file = str(RAW_POSTPAID_PANEL)
output_dir = str(MONTHLY_DIR)
os.makedirs(output_dir, exist_ok=True)
final_file = os.path.join(output_dir, "final_monthly_postpaid.parquet")

# # Columns to use
# use_columns = ["totalunits", "month_year", "contract_account_hashed", "contract_hashed", "Type", "rate_category"]

t0 = time()
print("[INFO] Loading data lazily...")

same_day = (
    pl.scan_parquet(parquet_file)
    .with_columns(
        pl.col("month_year")
        .str.strptime(pl.Date, "%Y-%m")
        .alias("transaction_date")
    )
    .group_by(["contract_hashed", "transaction_date"])
    .agg([
        pl.len().alias("n_rows"),
        pl.col("totalunits").alias("values"),
        pl.col("rate_category").unique().alias("rate_categories")
    ])
    .filter(pl.col("n_rows") > 1)
)

print(same_day.collect().head(50))
# -----------------------------
# Load data
# -----------------------------
df = pl.scan_parquet(parquet_file).filter(pl.col("totalunits") != 0)

# # Create a single unique ID
# df = df.with_columns([
#     pl.coalesce(["contract_account_hashed", "contract_hashed"]).alias("account_id")
# ])

# Parse month/year to timestamp
split_col = pl.col("month_year").str.split_exact("-", 1)
df = df.with_columns([
    split_col.struct.field("field_0").alias("year_str"),
    split_col.struct.field("field_1").str.zfill(2).alias("month_str")
])
df = df.with_columns([
    (pl.col("year_str") + "-" + pl.col("month_str")).str.strptime(pl.Date, "%Y-%m").alias("transaction_date")
])
df = df.drop(["year_str", "month_str"])

df = (
    df.group_by([
        "contract_hashed",
        "transaction_date"
    ])
    .agg([
        pl.col("totalunits").sum().alias("totalunits"),
        pl.col("Type").last().alias("Type"),
        pl.col("rate_category").last().alias("rate_category")
    ])
)

# Verify aggregation worked
check = (
    df.group_by(["contract_hashed", "transaction_date"])
      .len()
      .filter(pl.col("len") > 1)
      .collect()
)

print(f"Duplicate contract/date rows after aggregation: {check.height}")

# Sort accounts
df = df.sort(["contract_hashed", "transaction_date"]).with_columns(
    pl.col("totalunits").cast(pl.Float32)
)
# -----------------------------
# Daily aggregation
# -----------------------------
# -----------------------------
# STEP 1: Create transaction-level ordering per account
# -----------------------------
df = df.sort(["contract_hashed", "transaction_date"])

df = df.with_columns([
    pl.col("transaction_date")
      .shift(-1)
      .over("contract_hashed")
      .alias("next_transaction_date")
])


df = df.filter(pl.col("next_transaction_date").is_not_null())

df = df.with_columns([
    (pl.col("next_transaction_date") - pl.col("transaction_date"))
        .dt.total_days()
        .alias("days_between")
])

df = df.filter(pl.col("days_between") > 0)

df = df.with_columns(
    (pl.col("totalunits") / pl.col("days_between"))
    .alias("daily_kwh")
)

# df = df.with_columns([
#     pl.when(pl.col("days_between") <= 0)
#       .then(1.0)
#       .otherwise(pl.col("days_between"))
#       .alias("days_between_safe")
# ])

# df = df.with_columns([
#     (pl.col("totalunits") / pl.col("days_between_safe")).alias("daily_kwh")
# ])

# -----------------------------
# Expand across months (CRITICAL FIX)
# -----------------------------
df = df.with_columns([
    (pl.col("transaction_date").dt.year() * 12 + pl.col("transaction_date").dt.month() - 1).alias("start_month"),
    (pl.col("next_transaction_date").dt.year() * 12 + pl.col("next_transaction_date").dt.month() - 1).alias("end_month")
])

df = df.with_columns([
    pl.int_ranges(pl.col("start_month"), pl.col("end_month") + 1).alias("months_array")
]).explode("months_array")


# Convert back to actual month
df = df.with_columns([
    (pl.col("months_array") // 12).cast(pl.Int32).alias("year"),
    (pl.col("months_array") % 12 + 1).cast(pl.Int32).alias("month"),
    pl.datetime(
        (pl.col("months_array") // 12).cast(pl.Int32),
        (pl.col("months_array") % 12 + 1).cast(pl.Int32),
        1
    ).alias("month_start")
])

# -----------------------------
# Correct month slicing
# -----------------------------
df = df.with_columns([
    pl.col("month_start").dt.days_in_month().alias("month_days"),
    (pl.col("month_start") + pl.duration(days=pl.col("month_start").dt.days_in_month() - 1)).alias("month_end"),
])

df = df.with_columns([
    pl.max_horizontal(["transaction_date", "month_start"]).alias("period_start"),
    pl.min_horizontal(["next_transaction_date", "month_end"]).alias("period_end"),
])

df = df.with_columns([
    (pl.col("period_end") - pl.col("period_start"))
        .dt.total_days()
        .alias("days_in_period")
])

# df = df.with_columns([
#     pl.when(pl.col("days_in_period") <= 0)
#       .then(1.0)
#       .otherwise(pl.col("days_in_period"))
#       .alias("days_in_period_safe")
# ])

# # Final kWh allocation
# df = df.with_columns([
#     (pl.col("daily_kwh") * pl.col("days_in_period_safe")).alias("kwh")
# ])

df = df.with_columns([
    (pl.col("period_end") - pl.col("period_start"))
        .dt.total_days()
        .alias("days_in_period")
])

df = df.filter(pl.col("days_in_period") > 0)

df = df.with_columns([
    (pl.col("daily_kwh") * pl.col("days_in_period"))
        .alias("kwh")
])

# -----------------------------
# FINAL monthly aggregation
# -----------------------------
# monthly = (
#     df.group_by(["contract_hashed", "Type", "rate_category", "month_start"])
#     .agg([
#         pl.col("kwh").sum()
#     ])
#     .with_columns([
#         pl.col("month_start").dt.strftime("%Y-%m").alias("month_year")
#     ])
# )

monthly = (
    df.group_by([
        "contract_hashed",
        "Type",
        "month_start"
    ])
    .agg([
        pl.col("kwh").sum().alias("kwh")
    ])
    .with_columns([
        pl.col("month_start")
          .dt.strftime("%Y-%m")
          .alias("month_year")
    ])
)

print(monthly.describe())

duplicates = (
    monthly
    .group_by(["contract_hashed", "month_start"])
    .len()
    .filter(pl.col("len") > 1)
    .collect()
)

print(f"Duplicate contract-months after final aggregation: {duplicates.height}")

if duplicates.height > 0:
    print(duplicates.head(20))
    raise ValueError("Monthly postpaid panel is not unique by contract-month.")

print(
    monthly
    .select([
        (pl.col("kwh") < 0).sum().alias("negative_months"),
        (pl.col("kwh") == 0).sum().alias("zero_months"),
        pl.len().alias("total_months")
    ])
    .collect()
)

print(
 monthly
 .with_columns(pl.col("kwh"))
 .group_by("month_start")
 .agg([
 (pl.col("kwh") == 0).sum().alias("n_zero"),
 (pl.col("kwh") < 1e-3).sum().alias("n_near_zero"),
 pl.count().alias("n_rows")
 ])
 .sort("month_start")
 .collect()
)

for m in ["2019-09","2019-11"]:
    print(m)
    print(
    df.filter(pl.col("month_start").dt.strftime("%Y-%m")==m)
    .select(["contract_hashed","transaction_date","next_transaction_date","days_between","daily_kwh","days_in_period","kwh"])
    .sort("kwh")
    .head(50)
    .collect()
    )

print(
 df.select([
 pl.col("totalunits").sum().alias("total_totalunits"),
 pl.col("daily_kwh").min().alias("min_daily_kwh"),
 pl.col("days_between").min().alias("min_days_between"),
 pl.col("days_in_period").min().alias("min_days_in_period")
 ]).collect()
)

# print(df.select([pl.col("totalunits").dtype, pl.col("daily_kwh").dtype]).unique().collect())

print("negative totalunits:") 
print(pl.scan_parquet(parquet_file).filter(pl.col("totalunits") < 0).select(["contract_hashed","month_year","totalunits"]).limit(20).collect())

print("negative daily_kwh (after transforms):")
print(
    df
    .filter(pl.col("daily_kwh") < 0)
    .select(["contract_hashed","transaction_date","next_transaction_date","totalunits","days_between","daily_kwh"])
    .limit(50)
    .collect()
)

print(df.schema)  

monthly = monthly.filter(pl.col("kwh") > 0)


# Save monthly
monthly.collect().write_parquet(final_file)
print(f"[INFO] ✅ Final merged file saved: {final_file}")
print(f"[INFO] Total runtime: {time() - t0:.1f}s")

monthly_df = monthly.collect()


# -----------------------------
# Unique contracts (TOTAL)
# -----------------------------
n_unique_contracts = monthly_df.select("contract_hashed").n_unique()
print(f"Total unique contracts (contract_hashed): {n_unique_contracts}")

summary_table = (
    monthly_df
    .group_by(["month_year", "Type"])
    .agg([
        pl.col("kwh").mean().alias("mean_kwh"),
        pl.col("kwh").median().alias("median_kwh"),
        pl.col("kwh").quantile(0.1).alias("p10_kwh"),
        pl.col("kwh").quantile(0.9).alias("p90_kwh"),
        pl.len().alias("num_accounts")
    ])
    .sort("month_year")
)

print("\n[INFO] Monthly Consumption Distribution:")
# <- ensure eager
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
print(summary_table.to_pandas())


df_months = (
    monthly
    .select(["contract_hashed", "month_start"])
    .unique()
    .collect()
)

bounds = (
    df_months
    .group_by("contract_hashed")
    .agg([
        pl.col("month_start").min().alias("min_month"),
        pl.col("month_start").max().alias("max_month"),
    ])
)

expected = (
    bounds
    .with_columns([
        pl.int_ranges(
            (pl.col("min_month").dt.year() * 12 + pl.col("min_month").dt.month() - 1),
            (pl.col("max_month").dt.year() * 12 + pl.col("max_month").dt.month())
        ).alias("month_id")
    ])
    .explode("month_id")
    .with_columns([
        (pl.col("month_id") // 12).cast(pl.Int32).alias("year"),
        (pl.col("month_id") % 12 + 1).cast(pl.Int32).alias("month"),
        pl.datetime(
            (pl.col("month_id") // 12).cast(pl.Int32),
            (pl.col("month_id") % 12 + 1).cast(pl.Int32),
            1
        ).alias("month_start")
    ])
    .select(["contract_hashed", "month_start"])
)

missing = expected.join(
    df_months,
    on=["contract_hashed", "month_start"],
    how="anti"
)

missing_count = missing.height

print(f"[INFO] Missing constructed months: {missing_count}")

if missing_count == 0:
    print("[INFO] ✅ PERFECT: No gaps in constructed panel")
else:
    print("[WARNING] ❌ Gap exists in constructed panel")

print(
    missing
    .sort(["contract_hashed", "month_start"])
    .head(50)
)

print(
    missing
    .group_by("month_start")
    .len()
    .sort("month_start")
)