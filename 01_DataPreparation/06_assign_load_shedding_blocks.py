# # Merge dataset (contracts with transactions, location, and SHS data) with load shedding blocks

# import os 
# import time
# import pandas as pd
# import geopandas as gpd
# from shapely import wkt

# # ------------------------------
# # CONFIG
# # ------------------------------
# COMBINED_FILE = "/home/ey53/vscode-server-backup/CapeTown_Workflow/5c_out_04192026/with_sseg_reg.parquet"
# BLOCKS_FILE = "/home/ey53/vscode-server-backup/CapeTown_Workflow/Load_shedding_Blocks.geojson"
# OUTPUT_FILE = "/home/ey53/vscode-server-backup/CapeTown_Workflow/6_out_04192026/merged_with_blocks_combined.parquet"

# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# CHUNK_SIZE = 100_000

# # ------------------------------
# # Step 1 — Load load shedding blocks
# # ------------------------------
# print(f"📂 Loading load shedding blocks: {BLOCKS_FILE}")
# blocks_gdf = gpd.read_file(BLOCKS_FILE)
# blocks_gdf = blocks_gdf.to_crs("EPSG:4326")
# print(f"✅ Loaded {len(blocks_gdf):,} load shedding blocks.")
# print(blocks_gdf.columns)
# print(blocks_gdf.describe())

# # ------------------------------
# # Step 2 — Load combined parquet
# # ------------------------------
# print(f"\n📂 Loading combined parquet: {COMBINED_FILE}")
# df = pd.read_parquet(COMBINED_FILE)
# print(df.describe())
# total_rows = len(df)

# # ---------------------------
# # Unique contracts (overall)
# # ---------------------------
# total_unique_contracts = df['contract_account_hashed'].nunique()

# # ---------------------------
# # Unique contracts with Area_number NOT NaN
# # ---------------------------
# unique_with_area = df.loc[df['geometry_wkt'].notna(), 'contract_account_hashed'].nunique()

# print(f"Total rows: {total_rows}")
# print(f"Total unique contracts: {total_unique_contracts}")
# print(f"Unique contracts with geometry_wkt: {unique_with_area}")

# # print(f"✅ Loaded {total_rows:,} rows, {len(df.columns)} columns.")
# # print(df.columns)
# # print(df['geometry_wkt'].describe())

# processed_chunks = []
# start_time = time.time()

# # ------------------------------
# # Step 3 — Chunked spatial join
# # ------------------------------
# for start in range(0, total_rows, CHUNK_SIZE):
#     print(f"\n🟦 Processing chunk {start // CHUNK_SIZE + 1} — rows {start:,} to {min(start + CHUNK_SIZE, total_rows):,}")

#     chunk_df = df.iloc[start:start + CHUNK_SIZE].copy()
#     if chunk_df.empty:
#         continue

#     if "geometry_wkt" not in chunk_df.columns:
#         print("⚠ 'wkt' column missing — skipping chunk")
#         continue
#     print(chunk_df["geometry_wkt"].iloc[0])
    
#     # Convert WKT to geometry
#     chunk_df["geometry"] = chunk_df["geometry_wkt"].apply(wkt.loads)

#     # Avoid geometry name conflicts
#     if "geometry_block" in chunk_df.columns:
#         chunk_df = chunk_df.drop(columns=["geometry_block"])

#     chunk_gdf = gpd.GeoDataFrame(chunk_df, geometry="geometry", crs="EPSG:32734")
#     chunk_gdf = chunk_gdf.to_crs("EPSG:4326")
    
#     print(chunk_gdf.geom_type.value_counts())
#     print(blocks_gdf.geom_type.value_counts())
#     print("Chunk bounds:", chunk_gdf.total_bounds)
#     print("Blocks bounds:", blocks_gdf.total_bounds)
#     test = chunk_gdf.head(5)    
#     test_join = gpd.sjoin(test, blocks_gdf, predicate="intersects")
#     print(test_join)

#     # Spatial join
#     merged_chunk = gpd.sjoin(
#         chunk_gdf,
#         blocks_gdf,
#         how="left",
#         predicate="intersects",
#         rsuffix="_block"
#     )

#     processed_chunks.append(merged_chunk)

#     elapsed = time.time() - start_time
#     print(f"   ✅ Chunk processed in {elapsed:.1f}s")

# # ------------------------------
# # Step 4 — Save result
# # ------------------------------
# if processed_chunks:
#     result_gdf = pd.concat(processed_chunks, ignore_index=True)
#     result_gdf.to_parquet(OUTPUT_FILE, index=False)
#     print(f"\n💾 Saved merged file to: {OUTPUT_FILE}")
#     print(f"✅ Total merged rows: {len(result_gdf):,}")
# else:
#     print("⚠ No chunks processed — no output saved.")

# print(result_gdf['BlockID'].unique())
# print(result_gdf['BlockID'].describe())


# total_rows = len(result_gdf)

# # ---------------------------
# # Unique contracts (overall)
# # ---------------------------
# total_unique_contracts = result_gdf['contract_account_hashed'].nunique()

# # ---------------------------
# # Unique contracts with Area_number NOT NaN
# # ---------------------------
# unique_with_area = result_gdf.loc[result_gdf['geometry_wkt'].notna(), 'contract_account_hashed'].nunique()

# print(f"Total rows: {total_rows}")
# print(f"Total unique contracts: {total_unique_contracts}")
# print(f"Unique contracts with geometry_wkt: {unique_with_area}")


# Merge contract-month panel with load-shedding blocks.
#
# A spatial match is performed once per unique WKT location rather than once
# for every monthly observation. The resulting location-to-block lookup is then
# merged back into the full monthly panel.

from __future__ import annotations

import os
import time

import geopandas as gpd
import pandas as pd


###############################################################################
# PATHS
###############################################################################

COMBINED_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "5c_out_09132026/with_sseg_reg.parquet"
)

BLOCKS_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "Load_shedding_Blocks.geojson"
)

OUTPUT_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "6_out_09132026/merged_with_blocks_combined.parquet"
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)


###############################################################################
# HELPERS
###############################################################################

def section(title: str) -> None:
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


###############################################################################
# LOAD LOAD-SHEDDING BLOCKS
###############################################################################

section("LOADING LOAD-SHEDDING BLOCKS")

print(
    f"Loading load-shedding blocks: {BLOCKS_FILE}",
    flush=True,
)

blocks_gdf = gpd.read_file(BLOCKS_FILE)

print(
    "Original block CRS:",
    blocks_gdf.crs,
    flush=True,
)

if blocks_gdf.crs is None:
    raise ValueError(
        "Load-shedding block file has no CRS."
    )

blocks_gdf = blocks_gdf.to_crs("EPSG:4326")

print(
    "Blocks loaded:",
    f"{len(blocks_gdf):,}",
    flush=True,
)

print(
    "Block columns:",
    blocks_gdf.columns.tolist(),
    flush=True,
)

print(
    "Block geometry types:",
    flush=True,
)

print(
    blocks_gdf.geom_type.value_counts(),
    flush=True,
)

if "BlockID" not in blocks_gdf.columns:
    raise KeyError(
        "The block file does not contain a BlockID column."
    )

missing_block_geometry = int(
    blocks_gdf.geometry.isna().sum()
)

empty_block_geometry = int(
    blocks_gdf.geometry.is_empty.sum()
)

print(
    "Blocks with missing geometry:",
    f"{missing_block_geometry:,}",
    flush=True,
)

print(
    "Blocks with empty geometry:",
    f"{empty_block_geometry:,}",
    flush=True,
)

if missing_block_geometry > 0 or empty_block_geometry > 0:
    blocks_gdf = blocks_gdf.loc[
        blocks_gdf.geometry.notna()
        & ~blocks_gdf.geometry.is_empty
    ].copy()

# Retain only fields needed downstream.
block_columns = [
    "BlockID",
    "geometry",
]

blocks_lookup_gdf = blocks_gdf[
    block_columns
].copy()

duplicate_block_ids = (
    blocks_lookup_gdf["BlockID"]
    .duplicated(keep=False)
    .sum()
)

print(
    "Rows involving duplicated BlockID values:",
    f"{duplicate_block_ids:,}",
    flush=True,
)

# Duplicate BlockIDs are acceptable if one block consists of multiple polygon
# pieces, but dissolve them to one geometry per BlockID.
if duplicate_block_ids > 0:
    blocks_lookup_gdf = (
        blocks_lookup_gdf
        .dissolve(
            by="BlockID",
            as_index=False,
        )
    )

print(
    "Unique block IDs after preparation:",
    f"{blocks_lookup_gdf['BlockID'].nunique():,}",
    flush=True,
)


###############################################################################
# LOAD CONTRACT-MONTH PANEL
###############################################################################

section("LOADING CONTRACT-MONTH PANEL")

load_start = time.time()

df = pd.read_parquet(COMBINED_FILE)

print(
    "Panel loaded in seconds:",
    f"{time.time() - load_start:.1f}",
    flush=True,
)

required_columns = [
    "contract_account_hashed",
    "month_year",
    "wkt",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Combined panel is missing required columns: "
        f"{missing_columns}"
    )

total_rows = len(df)

total_unique_accounts = (
    df["contract_account_hashed"].nunique()
)

print(
    "Total panel rows:",
    f"{total_rows:,}",
    flush=True,
)

print(
    "Total unique accounts:",
    f"{total_unique_accounts:,}",
    flush=True,
)

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
    "Rows involved in duplicate account-months:",
    f"{input_duplicate_rows:,}",
    flush=True,
)

if input_duplicate_rows > 0:
    raise ValueError(
        "Input panel is not unique by account-month."
    )


###############################################################################
# PREPARE VALID WKT LOCATIONS
###############################################################################

section("PREPARING UNIQUE CONTRACT LOCATIONS")

valid_wkt = (
    df["wkt"].notna()
    & df["wkt"].astype(str).str.strip().ne("")
    & df["wkt"].astype(str).str.strip().ne("<NA>")
)

rows_with_valid_wkt = int(
    valid_wkt.sum()
)

accounts_with_valid_wkt = (
    df.loc[
        valid_wkt,
        "contract_account_hashed",
    ]
    .nunique()
)

print(
    "Rows with valid WKT:",
    f"{rows_with_valid_wkt:,}",
    flush=True,
)

print(
    "Unique accounts with valid WKT:",
    f"{accounts_with_valid_wkt:,}",
    flush=True,
)

location_lookup = (
    df.loc[
        valid_wkt,
        ["wkt"],
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

location_lookup["location_id"] = (
    location_lookup.index.astype("int64")
)

print(
    "Unique WKT locations:",
    f"{len(location_lookup):,}",
    flush=True,
)

print("\nChecking accounts with multiple locations...")

account_location_lookup = (
    df.loc[
        valid_wkt,
        [
            "contract_account_hashed",
            "wkt",
        ],
    ]
    .drop_duplicates()
)

account_location_counts = (
    account_location_lookup
    .groupby("contract_account_hashed")["wkt"]
    .nunique()
)

print(
    "Distribution of unique WKT locations per account:"
)

print(
    account_location_counts
    .value_counts()
    .sort_index()
)

multi_location_accounts = int(
    account_location_counts.gt(1).sum()
)

print(
    "Accounts with more than one unique WKT:",
    f"{multi_location_accounts:,}",
)

if multi_location_accounts > 0:
    example_accounts = (
        account_location_counts.loc[
            account_location_counts.gt(1)
        ]
        .head(10)
        .index
    )

    print(
        "\nExample accounts with multiple locations:"
    )

    print(
        df.loc[
            df["contract_account_hashed"].isin(
                example_accounts
            ),
            [
                "contract_account_hashed",
                "month_year",
                "wkt",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "contract_account_hashed",
                "month_year",
            ]
        )
        .to_string(index=False)
    )
    
###############################################################################
# CONVERT UNIQUE LOCATIONS TO GEOMETRY
###############################################################################

section("CONVERTING UNIQUE LOCATIONS TO GEOMETRY")

from shapely import wkt

location_geometry = location_lookup["wkt"].apply(
    lambda x: wkt.loads(x) if pd.notna(x) else None
)

invalid_geometry = location_geometry.apply(
    lambda g: (
        g is None
        or (hasattr(g, "is_empty") and g.is_empty)
    )
)

print(
    "Unique locations with invalid geometry:",
    f"{invalid_geometry.sum():,}",
    flush=True,
)

if invalid_geometry.any():
    print(
        "Examples of invalid WKT:",
        flush=True,
    )

    print(
        location_lookup.loc[
            invalid_geometry,
            "wkt",
        ].head(20).to_string(index=False),
        flush=True,
    )

location_gdf = gpd.GeoDataFrame(
    location_lookup.loc[
        ~invalid_geometry
    ].copy(),
    geometry=location_geometry.loc[
        ~invalid_geometry
    ],
    crs="EPSG:4326",
)

print(
    "Location geometry types:",
    flush=True,
)

print(
    location_gdf.geom_type.value_counts(),
    flush=True,
)

non_point_locations = (
    ~location_gdf.geom_type.eq("Point")
)

print(
    "Unique locations that are not points:",
    f"{non_point_locations.sum():,}",
    flush=True,
)

if non_point_locations.any():
    raise ValueError(
        "Contract WKT contains non-point geometries. "
        "Inspect these before assigning load-shedding blocks."
    )

print(
    "Contract-location bounds:",
    location_gdf.total_bounds,
    flush=True,
)

print(
    "Block bounds:",
    blocks_lookup_gdf.total_bounds,
    flush=True,
)


###############################################################################
# SPATIAL JOIN: LOCATION WITHIN BLOCK
###############################################################################

section("ASSIGNING LOCATIONS TO LOAD-SHEDDING BLOCKS")

join_start = time.time()

location_blocks = gpd.sjoin(
    location_gdf,
    blocks_lookup_gdf,
    how="left",
    predicate="within",
)

print(
    "Spatial join completed in seconds:",
    f"{time.time() - join_start:.1f}",
    flush=True,
)

print(
    "Rows before spatial join:",
    f"{len(location_gdf):,}",
    flush=True,
)

print(
    "Rows after spatial join:",
    f"{len(location_blocks):,}",
    flush=True,
)

multiple_matches = (
    location_blocks.duplicated(
        subset=["location_id"],
        keep=False,
    )
)

print(
    "Rows involved in multiple block matches:",
    f"{multiple_matches.sum():,}",
    flush=True,
)

print(
    "Distinct locations with multiple block matches:",
    location_blocks.loc[
        multiple_matches,
        "location_id",
    ].nunique(),
    flush=True,
)

if multiple_matches.any():
    print(
        "Multiple-match examples:",
        flush=True,
    )

    print(
        location_blocks.loc[
            multiple_matches,
            [
                "location_id",
                "wkt",
                "BlockID",
            ],
        ]
        .sort_values(
            [
                "location_id",
                "BlockID",
            ]
        )
        .head(100)
        .to_string(index=False),
        flush=True,
    )

    raise ValueError(
        "Some contract locations matched multiple load-shedding blocks."
    )

location_block_lookup = (
    location_blocks[
        [
            "wkt",
            "BlockID",
        ]
    ]
    .copy()
)

location_block_lookup["block_match_type"] = (
    location_block_lookup["BlockID"]
    .notna()
    .map(
        {
            True: "within",
            False: "unmatched",
        }
    )
)

# Add invalid WKT locations back as unmatched so every unique WKT key is
# represented in the lookup.
if invalid_geometry.any():
    invalid_lookup = (
        location_lookup.loc[
            invalid_geometry,
            ["wkt"],
        ]
        .copy()
    )

    invalid_lookup["BlockID"] = pd.NA
    invalid_lookup["block_match_type"] = (
        "invalid_geometry"
    )

    location_block_lookup = pd.concat(
        [
            location_block_lookup,
            invalid_lookup,
        ],
        ignore_index=True,
    )

lookup_duplicates = (
    location_block_lookup.duplicated(
        subset=["wkt"],
        keep=False,
    ).sum()
)

print(
    "Duplicate WKT keys in final block lookup:",
    f"{lookup_duplicates:,}",
    flush=True,
)

if lookup_duplicates > 0:
    raise ValueError(
        "The location-to-block lookup is not unique by WKT."
    )

print(
    "\nUnique-location block match counts:",
    flush=True,
)

print(
    location_block_lookup[
        "block_match_type"
    ].value_counts(dropna=False),
    flush=True,
)


###############################################################################
# MERGE BLOCK LOOKUP INTO FULL MONTHLY PANEL
###############################################################################

section("MERGING BLOCK ASSIGNMENTS INTO MONTHLY PANEL")

rows_before_merge = len(df)

df = df.merge(
    location_block_lookup,
    on="wkt",
    how="left",
    validate="many_to_one",
)

rows_after_merge = len(df)

print(
    "Rows before block merge:",
    f"{rows_before_merge:,}",
    flush=True,
)

print(
    "Rows after block merge:",
    f"{rows_after_merge:,}",
    flush=True,
)

if rows_after_merge != rows_before_merge:
    raise ValueError(
        "Block lookup merge changed the panel row count."
    )

df["block_match_type"] = (
    df["block_match_type"]
    .fillna("missing_wkt")
)

final_duplicate_rows = int(
    df.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Final rows involved in duplicate account-months:",
    f"{final_duplicate_rows:,}",
    flush=True,
)

if final_duplicate_rows > 0:
    raise ValueError(
        "Block merge created duplicate account-month rows."
    )


###############################################################################
# FINAL DIAGNOSTICS
###############################################################################

section("FINAL BLOCK-ASSIGNMENT DIAGNOSTICS")

print(
    "Monthly-row block match counts:",
    flush=True,
)

print(
    df["block_match_type"]
    .value_counts(dropna=False),
    flush=True,
)

print(
    "\nBlockID row distribution:",
    flush=True,
)

print(
    df["BlockID"]
    .value_counts(dropna=False)
    .sort_index(),
    flush=True,
)

unique_accounts_with_block = (
    df.loc[
        df["BlockID"].notna(),
        "contract_account_hashed",
    ]
    .nunique()
)

unique_accounts_without_block = (
    df.loc[
        df["BlockID"].isna(),
        "contract_account_hashed",
    ]
    .nunique()
)

print(
    "Unique accounts with assigned block:",
    f"{unique_accounts_with_block:,}",
    flush=True,
)

print(
    "Unique accounts without assigned block:",
    f"{unique_accounts_without_block:,}",
    flush=True,
)

accounts_with_multiple_blocks = (
    df.loc[
        df["BlockID"].notna(),
        [
            "contract_account_hashed",
            "BlockID",
        ],
    ]
    .drop_duplicates()
    .groupby(
        "contract_account_hashed"
    )["BlockID"]
    .nunique()
    .gt(1)
    .sum()
)

print(
    "Accounts assigned to multiple blocks over time:",
    f"{accounts_with_multiple_blocks:,}",
    flush=True,
)

# Multiple blocks over time can be legitimate if an account changes location.
# Confirm that a block never changes when the WKT remains constant.
block_changes_without_location_change = (
    df.loc[
        df["BlockID"].notna(),
        [
            "contract_account_hashed",
            "wkt",
            "BlockID",
        ],
    ]
    .drop_duplicates()
    .groupby(
        [
            "contract_account_hashed",
            "wkt",
        ]
    )["BlockID"]
    .nunique()
    .gt(1)
    .sum()
)

print(
    "Account-location pairs assigned to multiple blocks:",
    f"{block_changes_without_location_change:,}",
    flush=True,
)

if block_changes_without_location_change > 0:
    raise ValueError(
        "The same account-location pair was assigned "
        "to multiple load-shedding blocks."
    )


###############################################################################
# SAVE
###############################################################################

section("SAVING OUTPUT")

df.to_parquet(
    OUTPUT_FILE,
    engine="pyarrow",
    index=False,
)

print(
    "Saved merged file to:",
    OUTPUT_FILE,
    flush=True,
)

print(
    "Saved rows:",
    f"{len(df):,}",
    flush=True,
)

print(
    "Saved columns:",
    f"{len(df.columns):,}",
    flush=True,
)