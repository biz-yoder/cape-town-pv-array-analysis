
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import SHS_CLASSIFIED_PANEL, SHS_MATCHED_DIR
# Cleans SHS predictions based on constructed assumptions

import pandas as pd
import glob
import os
import numpy as np

# Paths
parquet_dir = str(SHS_MATCHED_DIR)
parquet_out = str(SHS_CLASSIFIED_PANEL.parent)

os.makedirs(parquet_out, exist_ok=True)

# Find all parquet files
# parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))

parquet_files = sorted(
    glob.glob(
        os.path.join(
            parquet_dir,
            "merged_contract_SHS_*.parquet"
        )
    )
)

print(f"📂 Found {len(parquet_files)} parquet files")

# Read and combine
dfs = []
for f in parquet_files:
    print(f"🔹 Loading {os.path.basename(f)} ...")
    df = pd.read_parquet(f)
    print(f"   → {len(df):,} rows, {len(df.columns)} columns")
    pred_cols = [c for c in df.columns if 'prediction' in c.lower()]

    if pred_cols:
        print(f"\n{os.path.basename(f)}")
        print(pred_cols)

    dfs.append(df)

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Combined DataFrame: {len(combined_df):,} total rows, {len(combined_df.columns)} columns")
else:
    raise ValueError("⚠️ No Parquet files found in the directory.")

# # Clean wkt
# combined_df = combined_df.loc[
#     combined_df["wkt"].notna() &
#     (combined_df["wkt"] != "") &
#     (combined_df["wkt"] != "<NA>") &
#     (combined_df["wkt"].str.len() > 10)
# ]

# Unique contracts
if "contract_ID" in combined_df.columns:
    unique_contracts = combined_df["contract_ID"].nunique()
    print(f"🔸 Total unique contracts: {unique_contracts:,}")
if "Type" in combined_df.columns:
    unique_by_type = combined_df.groupby("Type")["contract_ID"].nunique().reset_index()
    print("\n🔹 Unique contracts by Type:")
    for _, row in unique_by_type.iterrows():
        print(f"   {row['Type']}: {row['contract_ID']:,}")

# Ensure datetime and extract year
combined_df['month_year'] = pd.to_datetime(combined_df['month_year'], errors='coerce')
combined_df['year'] = combined_df['month_year'].dt.year
combined_df['shs_label_per_contract'] = np.where(combined_df['shs_area_per_contract'] > 0, 'PV_normal', '')

#######################################
# Collapse to contract-year

shs_years = (
    combined_df
    .groupby(['contract_ID', 'year'], as_index=False)
    .agg(
        has_shs=('shs_label_per_contract', lambda x: (x == "PV_normal").any()),
        shs_area_per_contract=('shs_area_per_contract', 'first'),  # make sure column exists
        shs_label_per_contract=('shs_label_per_contract', 'first')
    )
)

#######################################
# Create source flag (observed)
# -----------------------------
shs_years['shs_source'] = np.where(
    shs_years['has_shs'],
    'observed',
    pd.NA
)

# Keep a copy of original state
shs_years['has_shs_original'] = shs_years['has_shs']

#######################################
# Define function to implement SHS assumptions

def fix_shs_years(df):
    df = df.copy()

    # Years where SHS was directly observed.
    observed_years = set(
        df.loc[
            df["shs_source"].eq("observed"),
            "year",
        ]
    )

    # RULE 1: Delete a single isolated observed year,
    # except when the only observed year is 2023.
    if len(observed_years) == 1:
        only_year = next(iter(observed_years))

        if only_year != 2023:
            df.loc[
                :,
                [
                    "has_shs",
                    "shs_area_per_contract",
                    "shs_label_per_contract",
                    "shs_source",
                ],
            ] = [
                False,
                np.nan,
                np.nan,
                pd.NA,
            ]

        return df

    # RULE 2: Fill only one-year gaps between observed years.
    #
    # Examples:
    # 2018 and 2020 observed -> fill 2019.
    # 2018 and 2021 observed -> do not fill 2019 or 2020.
    # 2018, 2020, and 2022 observed -> fill 2019 and 2021.
    if observed_years:
        observed_years_sorted = sorted(
            observed_years
        )

        for previous_year, next_year in zip(
            observed_years_sorted[:-1],
            observed_years_sorted[1:],
        ):
            if next_year - previous_year == 2:
                missing_year = previous_year + 1
                year_mask = df["year"].eq(
                    missing_year
                )

                if year_mask.any():
                    df.loc[
                        year_mask,
                        "has_shs",
                    ] = True

                    df.loc[
                        year_mask,
                        "shs_source",
                    ] = "gap_filled"

    return df

#######################################
# Apply function at contract level

shs_years = (
    shs_years
    .groupby('contract_ID', group_keys=False)
    .apply(fix_shs_years) 
)

#######################################
# Merge back into monthly data

combined_df = combined_df.drop(columns=['shs_area_per_contract', 'shs_label_per_contract'], errors='ignore')

combined_df = combined_df.merge(
    shs_years[['contract_ID', 'year', 'has_shs', 'shs_area_per_contract', 'shs_label_per_contract', 'shs_source']],
    on=['contract_ID', 'year'],
    how='left'
)

# Treat empty labels and zero area as missing only for years
# that Rule 2 classified as gap-filled.
gap_filled_mask = combined_df["shs_source"].eq("gap_filled")

combined_df.loc[
    gap_filled_mask,
    "shs_label_per_contract",
] = np.nan

combined_df.loc[
    gap_filled_mask,
    "shs_area_per_contract",
] = np.nan

combined_df['shs_label_edit'] = combined_df['shs_label_per_contract']
combined_df['shs_area_m2_edit'] = combined_df['shs_area_per_contract']

#######################################
# Remove SHS info where SHS doesn't exist

combined_df.loc[~combined_df['has_shs'], ['shs_label_edit', 'shs_area_m2_edit']] = np.nan

#######################################
# Forward-fill SHS attributes within contract

combined_df = combined_df.sort_values(['contract_ID', 'month_year'])
combined_df[['shs_label_edit', 'shs_area_m2_edit']] = (
    combined_df
    .groupby('contract_ID')[['shs_label_edit', 'shs_area_m2_edit']]
    .ffill()
)

# Retain forward-filled attributes only in contract-years classified as SHS.
combined_df.loc[
    ~combined_df["has_shs"],
    [
        "shs_label_edit",
        "shs_area_m2_edit",
    ],
] = np.nan

#######################################
# Create binary imputed flag (imputed)

combined_df['shs_imputed'] = combined_df['shs_source'].isin(['gap_filled', 'forward_extended'])

#######################################
# Check

print(
    combined_df.groupby(['year', 'shs_source'])['contract_ID']
    .nunique()
)

# -----------------------------------------------------------------------------------
# Summary:
# - Each contract-year/month has a consistent SHS status.
# - A missing SHS year is filled only when it is the sole missing
#   year between two directly observed SHS years.
# - 2023 is populated if SHS existed in prior years (forward_extended).
# - Single-year SHS is only allowed in 2023.
# - shs_source and shs_imputed flags allow distinguishing observed vs gap-filled vs forward-extended data.
# -----------------------------------------------------------------------------------

combined_df['shs_label_edit'] = np.where(
    combined_df['has_shs'] & combined_df['shs_label_edit'].isna(),
    'PV_normal',
    combined_df['shs_label_edit']
)

# Count unique contracts per year
unique_contracts_per_year = combined_df.groupby('year')['contract_ID'].nunique().reset_index()
unique_contracts_per_year.rename(columns={'contract_ID': 'unique_contracts'}, inplace=True)
print("\n🔹 Unique contracts by year:")
print(unique_contracts_per_year)

# Filter for households with SHS
pv_df = combined_df[combined_df['shs_label_edit'] == "PV_normal"]

# Count unique contracts per year among PV households
unique_contracts_per_year = pv_df.groupby('year')['contract_ID'].nunique().reset_index()
unique_contracts_per_year.rename(columns={'contract_ID': 'unique_contracts'}, inplace=True)

print("\n🔹 Unique PV_normal contracts by year:")
print(unique_contracts_per_year)

months_per_contract = combined_df.groupby('contract_ID')['month_year'].nunique()
print(months_per_contract.describe())

# Save
output_file = os.path.join(parquet_out, "combined.parquet")
print("Final columns:", combined_df.columns)

combined_df.to_parquet(output_file, index=False)
print(f"\n✅ Saved combined parquet to {output_file}")

audit = (
    shs_years
    .assign(
        pv_added = (
            (~shs_years['has_shs_original']) &
            (shs_years['has_shs'])
        ),
        pv_removed = (
            (shs_years['has_shs_original']) &
            (~shs_years['has_shs'])
        )
    )
    .groupby('year')
    .agg(
        total_contracts=('contract_ID', 'nunique'),
        pv_original=('has_shs_original', 'sum'),
        pv_final=('has_shs', 'sum'),
        pv_added=('pv_added', 'sum'),
        pv_removed=('pv_removed', 'sum')
    )
    .reset_index()
)

audit['pct_added'] = audit['pv_added'] / audit['total_contracts']
audit['pct_removed'] = audit['pv_removed'] / audit['total_contracts']
audit['pct_removed_of_original_pv'] = (
    audit['pv_removed'] / audit['pv_original']
)

audit["net_change"] = audit["pv_final"] - audit["pv_original"]

print(audit)


# import pandas as pd
# import glob
# import os
# import numpy as np

# # ==============================
# # PATHS
# # ==============================
# os.makedirs(parquet_out, exist_ok=True)

# # ==============================
# # FIND ALL PARQUET FILES
# # ==============================
# parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
# print(f"📂 Found {len(parquet_files)} parquet files")

# # Only include contract-level files (preserves rows even if SHS is missing)
# parquet_files = [f for f in parquet_files if "contract_shs_summary" in f or "combined" in f]
# print(f"🔹 Loading {len(parquet_files)} contract-level parquet files")

# # ==============================
# # READ AND COMBINE
# # ==============================
# dfs = []
# for f in parquet_files:
#     print(f"🔸 Loading {os.path.basename(f)} ...")
#     df = pd.read_parquet(f)
#     print(f"   → {len(df):,} rows, {len(df.columns)} columns")
#     dfs.append(df)

# if not dfs:
#     raise ValueError("⚠️ No contract-level Parquet files found!")

# combined_df = pd.concat(dfs, ignore_index=True)
# print(f"\n✅ Combined DataFrame: {len(combined_df):,} total rows, {len(combined_df.columns)} columns")

# # ==============================
# # CLEAN WKT (optional)
# # ==============================
# if "wkt" in combined_df.columns:
#     combined_df = combined_df.loc[
#         combined_df["wkt"].notna() &
#         (combined_df["wkt"] != "") &
#         (combined_df["wkt"] != "<NA>") &
#         (combined_df["wkt"].str.len() > 10)
#     ]

# # ==============================
# # ENSURE DATETIME AND EXTRACT YEAR
# # ==============================
# combined_df['month_year'] = pd.to_datetime(combined_df['month_year'], errors='coerce')
# combined_df['year'] = combined_df['month_year'].dt.year

# # ==============================
# # ENSURE COLUMNS EXIST
# # ==============================
# if 'shs_area_m2' not in combined_df.columns:
#     combined_df['shs_area_m2'] = np.nan
# if 'shs_label' not in combined_df.columns:
#     combined_df['shs_label'] = pd.NA

# # ==============================
# # COLLAPSE TO CONTRACT-YEAR
# # ==============================
# shs_years = (
#     combined_df
#     .groupby(['contract_ID', 'year'], as_index=False)
#     .agg(
#         has_shs=('shs_label', lambda x: (x == "PV_normal").any()),
#         shs_area_m2=('shs_area_m2', 'first'),
#         shs_label=('shs_label', 'first')
#     )
# )

# # Create source flag (observed)
# shs_years['shs_source'] = np.where(shs_years['has_shs'], 'observed', pd.NA)
# shs_years['has_shs_original'] = shs_years['has_shs']

# # ==============================
# # FUNCTION FOR ASSUMPTIONS
# # ==============================
# def fix_shs_years(df):
#     observed_years = set(df.loc[df['shs_source'] == 'observed', 'year'])

#     # RULE 1: DELETE single isolated year (except 2023)
#     if len(observed_years) == 1:
#         only_year = next(iter(observed_years))
#         if only_year != 2023:
#             df[['has_shs', 'shs_area_m2', 'shs_label', 'shs_source']] = [False, np.nan, pd.NA, pd.NA]
#             return df

#     # RULE 2: Fill gaps between observed years
#     if observed_years:
#         min_y, max_y = min(observed_years), max(observed_years)
#         for y in range(min_y, max_y + 1):
#             if y not in observed_years:
#                 if y in df['year'].values:
#                     df.loc[df['year'] == y, ['has_shs', 'shs_source']] = [True, 'gap_filled']
#                 else:
#                     new_row = {
#                         'contract_ID': df['contract_ID'].iloc[0],
#                         'year': y,
#                         'has_shs': True,
#                         'shs_area_m2': np.nan,
#                         'shs_label': pd.NA,
#                         'shs_source': 'gap_filled',
#                         'has_shs_original': False
#                     }
#                     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

#     # RULE 3: Forward-extend into 2023
#     if 2023 not in observed_years and any(y in observed_years for y in [2020, 2021, 2022]):
#         if 2023 in df['year'].values:
#             df.loc[df['year'] == 2023, ['has_shs', 'shs_source']] = [True, 'forward_extended']
#         else:
#             new_row = {
#                 'contract_ID': df['contract_ID'].iloc[0],
#                 'year': 2023,
#                 'has_shs': True,
#                 'shs_area_m2': np.nan,
#                 'shs_label': pd.NA,
#                 'shs_source': 'forward_extended',
#                 'has_shs_original': False
#             }
#             df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

#     return df

# # ==============================
# # APPLY ASSUMPTIONS
# # ==============================
# shs_years = shs_years.groupby('contract_ID', group_keys=False).apply(fix_shs_years)

# # ==============================
# # MERGE BACK TO MONTHLY DATA
# # ==============================
# combined_df = combined_df.drop(columns=['shs_area_m2', 'shs_label'], errors='ignore')

# combined_df = combined_df.merge(
#     shs_years[['contract_ID', 'year', 'has_shs', 'shs_area_m2', 'shs_label', 'shs_source']],
#     on=['contract_ID', 'year'],
#     how='left'
# )

# combined_df['shs_label_edit'] = combined_df['shs_label']
# combined_df['shs_area_m2_edit'] = combined_df['shs_area_m2']

# # Remove SHS info where SHS doesn't exist
# combined_df.loc[~combined_df['has_shs'], ['shs_label_edit', 'shs_area_m2_edit']] = np.nan

# # Forward-fill SHS attributes within contract
# combined_df = combined_df.sort_values(['contract_ID', 'month_year'])
# combined_df[['shs_label_edit', 'shs_area_m2_edit']] = combined_df.groupby('contract_ID')[['shs_label_edit', 'shs_area_m2_edit']].ffill()

# # Create binary imputed flag
# combined_df['shs_imputed'] = combined_df['shs_source'].isin(['gap_filled', 'forward_extended'])

# # Fill missing labels with default if SHS exists
# combined_df['shs_label_edit'] = np.where(
#     combined_df['has_shs'] & combined_df['shs_label_edit'].isna(),
#     'PV_normal',
#     combined_df['shs_label_edit']
# )

# # ==============================
# # DIAGNOSTICS
# # ==============================
# unique_contracts_per_year = combined_df.groupby('year')['contract_ID'].nunique().reset_index()
# unique_contracts_per_year.rename(columns={'contract_ID': 'unique_contracts'}, inplace=True)
# print("\n🔹 Unique contracts by year:")
# print(unique_contracts_per_year)

# pv_df = combined_df[combined_df['shs_label_edit'] == "PV_normal"]
# unique_contracts_per_year = pv_df.groupby('year')['contract_ID'].nunique().reset_index()
# unique_contracts_per_year.rename(columns={'contract_ID': 'unique_contracts'}, inplace=True)
# print("\n🔹 Unique PV_normal contracts by year:")
# print(unique_contracts_per_year)

# # ==============================
# # SAVE FINAL PARQUET
# # ==============================
# output_file = os.path.join(parquet_out, "combined.parquet")
# combined_df.to_parquet(output_file, index=False)
# print(f"\n✅ Saved combined parquet to {output_file}")

# # ==============================
# # AUDIT TABLE
# # ==============================
# audit = (
#     shs_years
#     .assign(
#         pv_added = (~shs_years['has_shs_original']) & (shs_years['has_shs']),
#         pv_removed = (shs_years['has_shs_original']) & (~shs_years['has_shs'])
#     )
#     .groupby('year')
#     .agg(
#         total_contracts=('contract_ID', 'nunique'),
#         pv_original=('has_shs_original', 'sum'),
#         pv_final=('has_shs', 'sum'),
#         pv_added=('pv_added', 'sum'),
#         pv_removed=('pv_removed', 'sum')
#     )
#     .reset_index()
# )

# audit['pct_added'] = audit['pv_added'] / audit['total_contracts']
# audit['pct_removed'] = audit['pv_removed'] / audit['total_contracts']
# audit['pct_removed_of_original_pv'] = audit['pv_removed'] / audit['pv_original']

# print("\n🔹 Audit table:")
# print(audit)
