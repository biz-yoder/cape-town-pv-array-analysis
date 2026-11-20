
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import MONTHLY_DIR, RAW_PREPAID_DIR
import polars as pl
import os
from time import time

# Imports all postpaid data and prepaid data from March 2021 and earlier. 

from glob import glob

SCHEMA_CASTS = {
    "totalvalue": pl.Float64,
    "totalunits": pl.Float32,
    "trf_id": pl.Utf8,
    "trfname": pl.Utf8,
    "step0_tariff": pl.Float64,
    "step0_units": pl.Float64,
    "step1_tariff": pl.Float64,
    "step1_units": pl.Float64,
    "fbe_units": pl.Float64,
    "vat_value": pl.Float64,
    "vat_refund": pl.Float64,
    "debt1_amount": pl.Float64,
    "debt_2_amount": pl.Float64,
    "refund": pl.Float64,
    "revenuedebt": pl.Float64,
    "service_charge": pl.Float64,

    # FIXED: all booleans → strings
    "contract_account_present": pl.Utf8,
    "lis_key_present": pl.Utf8,
    "device_serial_number_present": pl.Utf8,
    "is_prepaid_contract_account": pl.Utf8,
    "has_active_electricity_contract": pl.Utf8,
    "transaction_and_contract_lis_aligned": pl.Utf8,
    "is_electricity_meter": pl.Utf8,
    "has_active_electricity_meter": pl.Utf8,

    "year": pl.Int64,
    "month": pl.Int64,
    "contract_account_hashed": pl.Utf8,
    # "transaction_timestamp": pl.Datetime("ns")
}

ALL_COLUMNS = list(SCHEMA_CASTS.keys()) + [
    "transaction_timestamp",   # 🔥 ADD THIS
    "meter_install", "meter_remove", "prepaid_install",
    "lis_key_hashed", "device_serial_number_hashed",
    "terminal_hashed", "group_trn_no_hashed",
    "receipt_no_hashed", "agr_id_hashed",
    "rdp_id_hashed", "meter_id_hashed", "shop_name_hashed"
]

# -----------------------------
# Paths
# -----------------------------
parquet_path = str(RAW_PREPAID_DIR)
output_dir = str(MONTHLY_DIR)
os.makedirs(output_dir, exist_ok=True)
############################################

use_columns = ["totalunits", "trfname", "transaction_timestamp", "contract_account_hashed"]
final_file = os.path.join(output_dir, "final_monthly_prepaid.parquet")

# -----------------------------
# Parameters
# -----------------------------
batch_size = 10_000  # number of accounts per batch

parquet_path = str(RAW_PREPAID_DIR)
files = glob(os.path.join(parquet_path, "**/*.parquet"), recursive=True)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    start = time()

    print("[INFO] Scanning all parquet files...")
    
    # # Lazy scan all parquets
    # df_lazy = (
    #     pl.scan_parquet(os.path.join(parquet_path, "*.parquet"), missing_columns="insert", extra_columns="ignore")
    #     .filter(~pl.col("trfname").fill_null("").str.contains("(?i)Small Power"))
    # )
    # # Parse timestamps and cast totalunits
    # df_lazy = df_lazy.with_columns([
    #     pl.col("transaction_timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
    #     pl.col("totalunits").cast(pl.Float32)
    # ])

    files = glob(os.path.join(parquet_path, "**/*.parquet"), recursive=True)

    lfs = []

    for f in files:
        lf = pl.scan_parquet(f)

        schema_names = lf.collect_schema().names()

        exprs = []
        for col in ALL_COLUMNS:
            dtype = SCHEMA_CASTS.get(col, pl.Utf8)

            if col in schema_names:
                exprs.append(pl.col(col).cast(dtype, strict=False))
            else:
                exprs.append(pl.lit(None).cast(dtype).alias(col))

        lf = lf.with_columns(exprs)

        # ✅ parse timestamp BEFORE select
        lf = lf.with_columns(
            pl.coalesce([
                pl.col("transaction_timestamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),

                pl.col("transaction_timestamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
            ]).alias("transaction_timestamp")
        )

        # ✅ NOW enforce column order
        lf = lf.select(ALL_COLUMNS)

        # filter
        lf = lf.filter(
            ~pl.col("trfname").fill_null("").str.contains("(?i)Small Power")
        )

        lfs.append(lf)

    df_lazy = pl.concat(lfs, how="vertical")
    
    # Get all unique accounts
    unique_accounts = (
        df_lazy
        .select("contract_account_hashed")
        .unique()
        .collect(engine="streaming")
    )["contract_account_hashed"]

    print(f"[INFO] Total unique accounts: {len(unique_accounts)}")
    
    batch_files = []
    allocatable_totalunits = 0.0

    for i in range(0, len(unique_accounts), batch_size):
        batch_accounts = unique_accounts[i:i+batch_size]
        print(f"[INFO] Processing accounts {i} to {i+len(batch_accounts)}...")
        
        # 1️⃣ Filter batch lazily, then collect to eager
        df_batch = df_lazy.filter(
            pl.col("contract_account_hashed").is_in(batch_accounts)
        ).collect(streaming=True)  # <-- materialize

        # FEB DUPLICATION ##############
        # -----------------------------
        # Duplicate Jan 2020 → Feb 2020 (RAW LEVEL)
        # -----------------------------

        # Ensure timestamp is datetime (should already be)
        df_batch = df_batch.with_columns(
            pl.col("transaction_timestamp").cast(pl.Datetime)
        )

        # 1. Extract Jan 2020 rows
        jan_raw = df_batch.filter(
            (pl.col("transaction_timestamp").dt.year() == 2020) &
            (pl.col("transaction_timestamp").dt.month() == 1)
        )

        # 2. Shift them forward by 1 month
        feb_raw = jan_raw.with_columns(
            pl.col("transaction_timestamp").dt.offset_by("1mo")
        )

        # 3. Append back
        df_batch = pl.concat([df_batch, feb_raw])

        ####################
        print(f"[DEBUG] Batch {i} deduplicated: {len(df_batch)} rows")
        print(df_batch.head(5))

        # -------------------------------------------------
        # Preserve original prepaid purchase behavior
        # -------------------------------------------------
        purchase_summary = (
            df_batch
            .filter(
                pl.col("contract_account_hashed").is_not_null()
                & pl.col("transaction_timestamp").is_not_null()
                & pl.col("totalunits").is_not_null()
            )
            .with_columns([
                pl.col("transaction_timestamp")
                .dt.truncate("1mo")
                .alias("month_start")
            ])
            .group_by([
                "contract_account_hashed",
                "month_start"
            ])
            .agg([
                # Actual number of individual prepaid purchases
                pl.len().alias("num_transactions"),

                # Actual credits purchased during that calendar month
                pl.col("totalunits")
                .sum()
                .alias("purchased_kwh"),

                pl.col("totalunits")
                .mean()
                .alias("mean_purchase_kwh"),

                pl.col("totalunits")
                .median()
                .alias("median_purchase_kwh"),

                pl.col("totalunits")
                .max()
                .alias("max_purchase_kwh"),

                pl.col("totalunits")
                .min()
                .alias("min_purchase_kwh"),

                pl.col("transaction_timestamp")
                .min()
                .alias("first_purchase_timestamp"),

                pl.col("transaction_timestamp")
                .max()
                .alias("last_purchase_timestamp")
            ])
        )

        # -----------------------------
        # Sum all transactions per contract per day
        # -----------------------------
        # Aggregate per day per account (ignore trfname)
        # df_batch = (
        #     df_batch.with_columns([
        #         pl.col("transaction_timestamp").dt.date().alias("transaction_date")
        #     ])
        #     .group_by(["contract_account_hashed", "transaction_date"])
        #     .agg([
        #         pl.col("totalunits").sum().alias("totalunits_daily"),
        #         pl.col("trfname")
        #         .filter((pl.col("trfname").is_not_null()) & (pl.col("trfname") != "EMPTY"))
        #         .first()
        #         .alias("trfname")
        #     ])
            
        #     .sort(["contract_account_hashed", "transaction_date"])
        # )

        df_batch = (
            df_batch
            .filter(
                pl.col("contract_account_hashed").is_not_null()
                & pl.col("transaction_timestamp").is_not_null()
                & pl.col("totalunits").is_not_null()
            )
            .with_columns([
                pl.col("transaction_timestamp")
                .dt.date()
                .alias("transaction_date")
            ])
            .group_by([
                "contract_account_hashed",
                "transaction_date"
            ])
            .agg([
                # Total credits purchased that day
                pl.col("totalunits")
                .sum()
                .alias("totalunits_daily"),

                # Preserve the number of purchases that day for diagnostics
                pl.len()
                .alias("n_purchases_daily"),

                pl.col("trfname")
                .filter(
                    pl.col("trfname").is_not_null()
                    & (pl.col("trfname") != "EMPTY")
                )
                .first()
                .alias("trfname")
            ])
            .sort([
                "contract_account_hashed",
                "transaction_date"
            ])
        )

        daily_duplicates = (
            df_batch
            .group_by([
                "contract_account_hashed",
                "transaction_date"
            ])
            .len()
            .filter(pl.col("len") > 1)
        )

        print(
            "Duplicate account/date rows after daily aggregation: "
            f"{daily_duplicates.height}"
        )

        if daily_duplicates.height > 0:
            print(daily_duplicates.head(20))
            raise ValueError(
                "Daily prepaid data are not unique by account-date."
            )

        # SORT AFTER UNIQUE
        df_batch = df_batch.sort(["contract_account_hashed", "transaction_date"])

        # df_batch = df_batch.with_columns([
        #     pl.col("transaction_date")
        #     .shift(-1)
        #     .over("contract_account_hashed")
        #     .alias("next_timestamp")
        # ])

        # print(df_batch.head(5))

        # print(f"[DEBUG] Next timestamps for first 5 rows:\n{df_batch.select(['transaction_date','next_timestamp']).head(5)}")

        # #4️⃣ Compute days_between, safe days, daily_rate (eager)
        # df_batch = df_batch.with_columns([
        #     (pl.col("next_timestamp") - pl.col("transaction_date")).dt.total_days().alias("days_between"),
        # ])

        # print(
        #     df_batch
        #     .filter(pl.col("days_between") <= 1)
        #     .select(["transaction_date","next_timestamp","days_between"])
        #     .sort("days_between")
        #     .head(20)
        # )

        # # # -----------------------------
        # # # Compute next_date and days_between
        # # # -----------------------------
        # # df_batch = df_batch.with_columns([
        # #     pl.col("transaction_date").shift(-1).over("contract_account_hashed").alias("next_date")
        # # ])

        # # df_batch = df_batch.with_columns([
        # #     ((pl.col("next_date") - pl.col("transaction_date")).dt.days()).alias("days_between")
        # # ])

        # # -----------------------------
        # # Daily rate logic: if next transaction is the next day, assign full kWh to this day
        # # -----------------------------
        # df_batch = df_batch.with_columns([
        #     pl.when(pl.col("days_between") == 1)
        #     .then(pl.col("totalunits_daily"))  # assign all kWh to this day
        #     .otherwise(pl.col("totalunits_daily") / pl.col("days_between"))  # otherwise spread over days
        #     .alias("daily_rate")
        # ])

        # print(f"[DEBUG] Daily rates for first 5 rows:\n{df_batch.select(['totalunits_daily','days_between','daily_rate']).head(5)}")
        # df_batch = df_batch.filter(pl.col("next_timestamp").is_not_null())
        
        df_batch = df_batch.with_columns([
            pl.col("transaction_date")
            .shift(-1)
            .over("contract_account_hashed")
            .alias("next_timestamp")
        ])

        df_batch = df_batch.with_columns([
            (
                pl.col("next_timestamp")
                - pl.col("transaction_date")
            )
            .dt.total_days()
            .alias("days_between")
        ])

        # Retain only intervals that can actually be allocated
        df_batch = df_batch.filter(
            pl.col("next_timestamp").is_not_null()
            & (pl.col("days_between") > 0)
        )

        # Track the amount that should be recovered by the allocation
        batch_allocatable_total = (
            df_batch
            .select(
                pl.col("totalunits_daily").sum()
            )
            .item()
        )

        if batch_allocatable_total is not None:
            allocatable_totalunits += batch_allocatable_total

        df_batch = df_batch.with_columns([
            (
                pl.col("totalunits_daily")
                / pl.col("days_between")
            )
            .alias("daily_rate")
        ])
        # 3️⃣ Now you can safely do month expansion, period calculation, kwh
        df_batch = df_batch.with_columns([
            (pl.col("transaction_date").dt.year() * 12 + pl.col("transaction_date").dt.month() - 1).alias("start_month"),
            (pl.col("next_timestamp").dt.year() * 12 + pl.col("next_timestamp").dt.month() - 1).alias("end_month")
        ]).with_columns([
            pl.int_ranges(pl.col("start_month"), pl.col("end_month") + 1).alias("months_array")
        ]).explode("months_array")

        # 4️⃣ Compute month_start, year, month
        df_batch = df_batch.with_columns([
            (pl.col("months_array") // 12).cast(pl.Int32).alias("year"),
            (pl.col("months_array") % 12 + 1).cast(pl.Int32).alias("month"),
            pl.datetime(
                (pl.col("months_array") // 12).cast(pl.Int32),
                (pl.col("months_array") % 12 + 1).cast(pl.Int32),
                1
            ).alias("month_start")
        ])

        print(f"[DEBUG] Month expansion check:\n{df_batch.select(['transaction_date','next_timestamp','month_start']).head(10)}")

        # # Compute month slice boundaries and kWh allocation correctly
        # df_batch = df_batch.with_columns([
        #     pl.col("month_start").dt.days_in_month().alias("month_days"),
        #     (pl.col("month_start") + pl.duration(days=pl.col("month_start").dt.days_in_month() - 1)).alias("month_end"),
        # ])

        # df_batch = df_batch.with_columns([
        #     pl.max_horizontal(["transaction_date", "month_start"]).alias("period_start"),
        #     pl.min_horizontal(["next_timestamp", "month_end"]).alias("period_end"),
        # ])

        # df_batch = df_batch.with_columns([
        #     (pl.col("period_end") - pl.col("period_start"))
        #         .dt.total_days()
        #         .alias("days_in_period")
        # ])


        # df_batch = df_batch.with_columns([
        #     pl.when(pl.col("days_in_period") <= 0)
        #     .then(1.0)
        #     .otherwise(pl.col("days_in_period"))
        #     .alias("days_in_period_safe")
        # ])

        # # df_batch = df_batch.filter(pl.col("days_between").is_not_null())

        # # print(df_batch.select("days_between").describe())

        # df_batch = df_batch.with_columns([
        #     (pl.col("daily_rate") * pl.col("days_in_period_safe"))
        #         .alias("kwh")
        # ])

        # -------------------------------------------------
        # Slice each purchase interval across calendar months
        # Intervals are treated as [start, end)
        # -------------------------------------------------
        df_batch = df_batch.with_columns([
            pl.col("month_start")
            .dt.offset_by("1mo")
            .alias("next_month_start")
        ])

        df_batch = df_batch.with_columns([
            pl.max_horizontal([
                "transaction_date",
                "month_start"
            ]).alias("period_start"),

            pl.min_horizontal([
                "next_timestamp",
                "next_month_start"
            ]).alias("period_end")
        ])

        df_batch = df_batch.with_columns([
            (
                pl.col("period_end")
                - pl.col("period_start")
            )
            .dt.total_days()
            .alias("days_in_period")
        ])

        # Remove the ending-month row when the interval ends
        # exactly at the start of that month
        df_batch = df_batch.filter(
            pl.col("days_in_period") > 0
        )

        df_batch = df_batch.with_columns([
            (
                pl.col("daily_rate")
                * pl.col("days_in_period")
            )
            .alias("kwh")
        ])


        df_batch = df_batch.with_columns(
            pl.when(pl.col("trfname") == "EMPTY")
            .then(None)
            .otherwise(pl.col("trfname"))
            .alias("trfname")
        )
        print(f"[DEBUG] Period kWh check (first 10 rows):\n{df_batch.select(['transaction_date','next_timestamp','month_start','period_start','period_end','days_in_period','kwh']).tail(24)}")

        # Aggregate monthly
        # monthly_batch_lazy = df_batch.group_by(["contract_account_hashed", "month_start"]).agg([
        #     pl.col("kwh").sum(),
        #     pl.count().alias("num_transactions"),
        #     pl.col("trfname").drop_nulls().first().alias("trfname")
        # ]).with_columns([
        #     pl.col("month_start").dt.strftime("%Y-%m").alias("month_year")
        # ]).select(["contract_account_hashed", "trfname", "month_year", "kwh", "num_transactions"])
        # Monthly interval-based consumption allocation
        monthly_kwh = (
            df_batch
            .group_by([
                "contract_account_hashed",
                "month_start"
            ])
            .agg([
                pl.col("kwh")
                .sum()
                .alias("kwh"),

                pl.col("trfname")
                .drop_nulls()
                .first()
                .alias("trfname")
            ])
        )

        # Join actual transaction behavior to the allocated panel
        monthly_batch_lazy = (
            monthly_kwh
            .join(
                purchase_summary,
                on=[
                    "contract_account_hashed",
                    "month_start"
                ],
                how="full",
                coalesce=True
            )
            .with_columns([
                pl.col("month_start")
                .dt.strftime("%Y-%m")
                .alias("month_year"),

                # Months created by allocation may contain no actual purchase
                pl.col("num_transactions")
                .fill_null(0),

                pl.col("purchased_kwh")
                .fill_null(0.0)
            ])
            .select([
                "contract_account_hashed",
                "trfname",
                "month_start",
                "month_year",

                # Interval-based consumption proxy
                "kwh",

                # Actual purchase behavior
                "num_transactions",
                "purchased_kwh",
                "mean_purchase_kwh",
                "median_purchase_kwh",
                "min_purchase_kwh",
                "max_purchase_kwh",
                "first_purchase_timestamp",
                "last_purchase_timestamp"
            ])
        )
        
        print(f"[DEBUG] Aggregated monthly kWh (first 5 rows):\n{monthly_batch_lazy.tail(24)}")

        monthly_duplicates = (
            monthly_batch_lazy
            .group_by([
                "contract_account_hashed",
                "month_start"
            ])
            .len()
            .filter(pl.col("len") > 1)
        )

        print(
            "Duplicate account-month rows in batch: "
            f"{monthly_duplicates.height}"
        )

        if monthly_duplicates.height > 0:
            print(monthly_duplicates.head(20))
            raise ValueError(
                "Monthly prepaid data are not unique by account-month."
            )

        # Save batch
        batch_file = os.path.join(output_dir, f"monthly_batch_{i}.parquet")
        monthly_batch_lazy.write_parquet(batch_file)
        batch_files.append(batch_file)
        print(f"[INFO] Saved batch: {batch_file}")
    
        avg_monthly = monthly_batch_lazy.group_by("month_year").agg(pl.col("kwh").mean())

        print("\nAverage kWh by month_year:")
        print(avg_monthly)

    # -----------------------------
    # Combine all batches
    # -----------------------------
    print("[INFO] Combining all batches...")
    all_monthly = pl.concat([
        pl.read_parquet(f)
        for f in batch_files
    ])

    final_duplicates = (
        all_monthly
        .group_by([
            "contract_account_hashed",
            "month_start"
        ])
        .len()
        .filter(pl.col("len") > 1)
    )

    print(
        "Duplicate account-month rows after combining batches: "
        f"{final_duplicates.height}"
    )

    if final_duplicates.height > 0:
        print(final_duplicates.head(20))
        raise ValueError(
            "Combined prepaid panel is not unique by account-month."
        )

    all_monthly.write_parquet(final_file)
    
    print(f"[INFO] ✅ Final merged file saved: {final_file}")
    print(f"[INFO] Total runtime: {time() - start:.1f}s")


#######################################
# Final sanity check
# raw_total = (
#     pl.scan_parquet(os.path.join(parquet_path, "*.parquet"), extra_columns="ignore", missing_columns='insert')
#     .filter(~pl.col("trfname").str.contains("Small Power"))  # exclude Small Power
#     .select(pl.col("totalunits").sum())
#     .collect(streaming=True)[0,0]
# )

# processed_kwh = pl.scan_parquet(final_file).select(pl.col("kwh").sum()).collect(streaming=True)[0,0]

# diff = raw_total - processed_kwh
# diff_pct = diff / raw_total * 100

# print("\n[INFO] Sanity Check:")
# print(f"Raw totalunits: {raw_total:,.2f}")
# print(f"Processed kWh:  {processed_kwh:,.2f}")
# print(f"Difference: {diff:,.2f} ({diff_pct:.5f}%)")

processed_kwh = (
    pl.scan_parquet(final_file)
    .select(
        pl.col("kwh").sum()
    )
    .collect(engine="streaming")
    .item()
)

diff = allocatable_totalunits - processed_kwh

diff_pct = (
    diff / allocatable_totalunits * 100
    if allocatable_totalunits != 0
    else float("nan")
)

print("\n[INFO] Allocation Sanity Check:")
print(
    f"Allocatable purchased units: "
    f"{allocatable_totalunits:,.2f}"
)
print(
    f"Allocated monthly kWh:       "
    f"{processed_kwh:,.2f}"
)
print(
    f"Difference:                  "
    f"{diff:,.2f} ({diff_pct:.8f}%)"
)

avg_monthly = (
    pl.scan_parquet(final_file)
    .group_by(["month_year", "contract_account_hashed"])
    .agg(
        pl.col("kwh").sum().alias("monthly_kwh_per_account")
    )
    .group_by("month_year")
    .agg(
        pl.col("monthly_kwh_per_account").mean().alias("avg_kwh_per_account")
    )
    .sort("month_year")
    .collect(streaming=True)
)

print("\nAverage kWh by month_year:")
print(avg_monthly)

summary_table = (
    pl.scan_parquet(final_file)
    .group_by(["month_year"])
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

import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
print("SUMMARY TABLE")
print(summary_table.collect().to_pandas())


if abs(diff_pct) > 0.05:
    print("[WARNING] ⚠️ Significant discrepancy detected!")
else:
    print("[INFO] ✅ Sanity check passed!")
    print(f"[INFO] Total runtime: {time() - start:.1f}s")

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load monthly data
# -----------------------------
file = final_file
df = pl.read_parquet(file)

# -----------------------------
# Unique contracts (TOTAL)
# -----------------------------
n_unique_contracts = df.select("contract_account_hashed").n_unique()
print(f"Total unique contracts (contract_account_hashed): {n_unique_contracts}")

# Make sure month_year is datetime
df = df.with_columns([
    pl.col("month_year").str.strptime(pl.Date, "%Y-%m")
])

# -----------------------------
# Sort by account and month
# -----------------------------
df = df.sort(["contract_account_hashed", "month_year"])

# -----------------------------
# Compute previous month per contract
# -----------------------------
df = df.with_columns([
    pl.col("month_year").shift(1).over("contract_account_hashed").alias("prev_month")
])

# -----------------------------
# Compute month gap
# -----------------------------
df = df.with_columns([
    ((pl.col("month_year").dt.year() - pl.col("prev_month").dt.year()) * 12 +
     (pl.col("month_year").dt.month() - pl.col("prev_month").dt.month())
    ).alias("month_gap")
])

# Flag gaps > 1
df = df.with_columns([
    (pl.col("month_gap") > 1).alias("has_gap")
])

# Contracts with gaps
contracts_with_gaps = df.filter(pl.col("has_gap")).select("contract_account_hashed").unique()
print(f"Number of contracts with gaps: {contracts_with_gaps.height}")

# -----------------------------
# Missing months for each gap
# -----------------------------
gap_rows = df.filter(pl.col("has_gap")).to_pandas()  # easier to handle ranges in pandas

missing_months_list = []
for _, row in gap_rows.iterrows():
    # generate all missing months between prev_month and current month
    missing_range = pd.date_range(
        start=row['prev_month'] + pd.offsets.MonthBegin(1),
        end=row['month_year'] - pd.offsets.MonthBegin(1),
        freq='MS'
    )
    for m in missing_range:
        missing_months_list.append({
            'contract_account_hashed': row['contract_account_hashed'],
            'missing_month': m
        })

if len(missing_months_list) == 0:
    print("[INFO] No missing months detected. Skipping missing-month aggregation.")
else:
    missing_df = pd.DataFrame(missing_months_list)
    
    missing_counts = (
        missing_df.groupby('missing_month')['contract_account_hashed']
        .nunique()
        .sort_index()
    )
    
    print(missing_counts)

print(
    df.select(
        pl.col("days_between")
          .quantile([0.5,0.9,0.95,0.99,0.999])
    ).collect()
)