# Merges contract location data with building data from overture maps

from shapely import wkt
import geopandas as gpd
import pandas as pd
import os


# Paths
merged_path = "/home/ey53/vscode-server-backup/CapeTown_Workflow/2b_out_08032026/out_contract_with_location.parquet"
buildings_path = "/home/ey53/vscode-server-backup/CapeTown_Workflow/capetown_buildings2.parquet"
output_path = "/home/ey53/vscode-server-backup/CapeTown_Workflow/3_out_08032026/out_contractlocation_with_building.parquet"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

#######################################
# Load contract location+transactions data and filter WKT

merged_df = pd.read_parquet(merged_path)

# Total unique contracts before filtering
total_unique_contracts_all = merged_df['contract_ID'].nunique()
print(f"🔢 Total unique contracts (ALL, before WKT filter): {total_unique_contracts_all:,}")

# Unique contracts by Type
unique_contracts_by_type = (
    merged_df.groupby("Type")["contract_ID"].nunique().reset_index(name="unique_contracts")
)
print("Unique contracts by Type:")
print(unique_contracts_by_type.to_string(index=False))

# Keep only rows with valid WKT
merged_df = merged_df.loc[
    merged_df["wkt"].notna() &
    (merged_df["wkt"] != "") &
    (merged_df["wkt"] != "<NA>") &
    (merged_df["wkt"].str.len() > 10)
]
print(f"[Step 1] Rows with WKT: {len(merged_df):,}")

# Count unique contracts with WKT by Type
contracts_with_wkt_by_type = (
    merged_df.groupby("Type")["contract_ID"].nunique().reset_index(name="contracts_with_wkt")
)
print("Unique contracts with WKT by Type:")
print(contracts_with_wkt_by_type.to_string(index=False))

#######################################
# Convert WKT to geometry

merged_df["wkt"] = merged_df["wkt"].astype(str)
merged_gdf = gpd.GeoDataFrame(
    merged_df,
    geometry=gpd.GeoSeries.from_wkt(merged_df["wkt"]),
    crs="EPSG:4326"
)
print(f"✅ Geometry conversion successful. Rows remaining: {len(merged_gdf):,}")

#######################################
# Load building data

buildings_gdf = gpd.read_parquet(buildings_path)
print(f"[Step 2] Loaded buildings_gdf with {len(buildings_gdf):,} rows")

# Ensure CRS match
if buildings_gdf.crs != merged_gdf.crs:
    buildings_gdf = buildings_gdf.to_crs(merged_gdf.crs)
    print(f"[Step 3] Reprojected buildings_gdf to {merged_gdf.crs}")

#######################################
#  Spatial join: put contract in building

# # Filter buildings to bounding box
# minx, miny, maxx, maxy = merged_gdf.total_bounds
# buildings_subset = buildings_gdf.cx[minx:maxx, miny:maxy][['id', 'geometry']]

# joined_gdf = gpd.sjoin(
#     merged_gdf,
#     buildings_subset,
#     how="left",
#     predicate="within",
#     rsuffix="_building" #does this allow multiple household rows (bc diff tariffs) to match to a building
# )
# print(f"[Step 4] Spatial join complete — rows: {len(joined_gdf):,}")

#######################################
# Create unique contract-location lookup
#######################################

location_lookup = (
    merged_df[
        [
            "contract_ID",
            "Type",
            "wkt",
        ]
    ]
    .drop_duplicates()
    .copy()
)

print(
    "Unique contract-location combinations:",
    f"{len(location_lookup):,}"
)

location_gdf = gpd.GeoDataFrame(
    location_lookup,
    geometry=gpd.GeoSeries.from_wkt(
        location_lookup["wkt"]
    ),
    crs="EPSG:4326",
)

#######################################
# Prepare building data
#######################################

# Filter buildings to the bounds of the contract locations
minx, miny, maxx, maxy = location_gdf.total_bounds

buildings_subset = (
    buildings_gdf.cx[minx:maxx, miny:maxy][
        ["id", "geometry"]
    ]
    .copy()
    .rename(columns={"id": "building_id"})
)

#######################################
# Spatial join: assign containing building
#######################################

joined_locations = gpd.sjoin(
    location_gdf,
    buildings_subset,
    how="left",
    predicate="within",
)

print(
    "[Step 4] Spatial join complete — rows:",
    f"{len(joined_locations):,}"
)

#######################################
# Check for multiple containing buildings
#######################################

location_key = [
    "contract_ID",
    "wkt",
]

multiple_match_rows = joined_locations.duplicated(
    subset=location_key,
    keep=False,
)

print(
    "Rows involved in multiple building matches:",
    f"{multiple_match_rows.sum():,}"
)

print(
    "Distinct contract-location combinations with "
    "multiple building matches:",
    joined_locations.loc[
        multiple_match_rows,
        location_key,
    ].drop_duplicates().shape[0],
)

# Temporarily retain one building per contract-location.
# Inspect multiple-match cases before treating this as final.
joined_locations = (
    joined_locations
    .sort_values(location_key + ["building_id"])
    .drop_duplicates(
        subset=location_key,
        keep="first",
    )
)

#######################################
# Identify locations without a containing building
#######################################

contracts_unassigned = joined_locations.loc[
    joined_locations["building_id"].isna()
].copy()

num_unique_contracts = (
    contracts_unassigned["contract_ID"].nunique()
)

print(
    "Contracts without a containing building:",
    f"{num_unique_contracts:,}"
)

#######################################
# Assign nearest building within 15 meters
#######################################

# Give every unmatched contract-location row a stable identifier.
contracts_unassigned = (
    contracts_unassigned
    .reset_index(drop=True)
    .reset_index()
    .rename(columns={"index": "location_row_id"})
)

# Reproject points and buildings to a metric CRS.
contracts_unassigned_metric = (
    contracts_unassigned.to_crs(epsg=32734)
)

buildings_subset_metric = (
    buildings_subset.to_crs(epsg=32734)
)

# Remove spatial-join index columns before running another spatial join
contracts_unassigned_metric = contracts_unassigned_metric.drop(
    columns=["index_right"],
    errors="ignore",
)

buildings_subset_metric = buildings_subset_metric.drop(
    columns=["index_left", "index_right"],
    errors="ignore",
)

nearest = gpd.sjoin_nearest(
    contracts_unassigned_metric,
    buildings_subset_metric,
    how="left",
    distance_col="dist_m",
    max_distance=15,
)

print("\nNearest-building distance summary:")
print(
    nearest["dist_m"].describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

print("\nColumns returned by nearest join:")
print(nearest.columns.tolist())

#######################################
# Resolve equal-distance nearest-building ties
#######################################

nearest_duplicate_rows = nearest.duplicated(
    subset=["location_row_id"],
    keep=False,
)

print(
    "Rows involved in nearest-building ties:",
    f"{nearest_duplicate_rows.sum():,}"
)

print(
    "Distinct contract-location rows with nearest-building ties:",
    nearest.loc[
        nearest_duplicate_rows,
        ["location_row_id"]
    ].drop_duplicates().shape[0],
)

# Choose one nearest building per location.
# Sort first by distance and then by building_id for a deterministic tie-break.
# Identify the building-ID column created by the nearest join
if "building_id_right" in nearest.columns:
    nearest_building_col = "building_id_right"
elif "building_id" in nearest.columns:
    nearest_building_col = "building_id"
else:
    raise KeyError(
        "No building ID column found after nearest join. "
        f"Available columns: {nearest.columns.tolist()}"
    )

nearest_resolved = (
    nearest
    .sort_values(
        [
            "location_row_id",
            "dist_m",
            nearest_building_col,
        ]
    )
    .drop_duplicates(
        subset=["location_row_id"],
        keep="first",
    )
)

nearest_map = (
    nearest_resolved[
        [
            "location_row_id",
            nearest_building_col,
            "dist_m",
        ]
    ]
    .rename(
        columns={
            nearest_building_col: "building_id_nearest"
        }
    )
    .copy()
)

#######################################
# Add nearest assignments to unmatched locations
#######################################

contracts_unassigned = contracts_unassigned.merge(
    nearest_map,
    on="location_row_id",
    how="left",
    suffixes=("", "_nearest"),
    validate="one_to_one",
)

# Retain the nearest building where one was found.
contracts_unassigned["building_id"] = (
    contracts_unassigned["building_id_nearest"]
    .combine_first(
        contracts_unassigned["building_id"]
    )
)

contracts_unassigned["building_match_type"] = (
    contracts_unassigned["building_id"]
    .notna()
    .map({
        True: "nearest",
        False: "unmatched",
    })
)

contracts_unassigned = contracts_unassigned.drop(
    columns=[
        "building_id_nearest",
        "location_row_id",
        "index_right",
    ],
    errors="ignore",
)

#######################################
# Mark direct within-building matches
#######################################

joined_locations["building_match_type"] = (
    joined_locations["building_id"]
    .notna()
    .map({
        True: "within",
        False: "unmatched",
    })
)

#######################################
# Update joined location lookup with nearest assignments
#######################################

nearest_assignments = (
    contracts_unassigned[
        [
            "contract_ID",
            "wkt",
            "building_id",
            "building_match_type",
            "dist_m",
        ]
    ]
    .drop_duplicates(
        subset=[
            "contract_ID",
            "wkt",
        ]
    )
)

# Remove unmatched placeholder rows before adding the resolved nearest rows.
joined_locations = joined_locations.loc[
    joined_locations["building_id"].notna()
].copy()

joined_locations["dist_m"] = 0.0

joined_locations = pd.concat(
    [
        joined_locations,
        nearest_assignments,
    ],
    ignore_index=True,
    sort=False,
)

print("\nBuilding match type counts:")
print(joined_locations["building_match_type"].value_counts())

#######################################
# Confirm one building assignment per contract-location
#######################################

location_assignment_duplicates = joined_locations.duplicated(
    subset=[
        "contract_ID",
        "wkt",
    ],
    keep=False,
)

print(
    "\nContract-location rows involved in duplicate assignments:",
    f"{location_assignment_duplicates.sum():,}"
)

if location_assignment_duplicates.any():
    print(
        joined_locations.loc[
            location_assignment_duplicates,
            [
                "contract_ID",
                "wkt",
                "building_id",
                "building_match_type",
                "dist_m",
            ]
        ]
        .sort_values(
            [
                "contract_ID",
                "wkt",
                "dist_m",
            ]
        )
        .head(50)
    )

    raise ValueError(
        "Some contract-location combinations have multiple "
        "building assignments."
    )

#######################################
# Create one building lookup per contract-location
#######################################

building_lookup = (
    joined_locations[
        [
            "contract_ID",
            "wkt",
            "building_id",
            "building_match_type",
            "dist_m",
        ]
    ]
    .drop_duplicates(
        subset=[
            "contract_ID",
            "wkt",
        ]
    )
)

print(
    "Unique contract-location building assignments:",
    f"{len(building_lookup):,}"
)

#######################################
# Merge building assignments back to full monthly panel
#######################################

rows_before_building_merge = len(merged_df)

joined_df = merged_df.merge(
    building_lookup,
    on=[
        "contract_ID",
        "wkt",
    ],
    how="left",
    validate="many_to_one",
)

print(
    "Rows before building lookup merge:",
    f"{rows_before_building_merge:,}"
)

print(
    "Rows after building lookup merge:",
    f"{len(joined_df):,}"
)

if len(joined_df) != rows_before_building_merge:
    raise ValueError(
        "Building lookup merge changed the number of monthly rows."
    )

#######################################
# Find remaining unmatched contracts
#######################################

unmatched_unique_contracts = (
    joined_df.loc[
        joined_df["building_id"].isna(),
        "contract_ID",
    ]
    .nunique()
)

print(
    "Remaining unmatched UNIQUE contracts:",
    f"{unmatched_unique_contracts:,}"
)

unmatched_by_type = (
    joined_df.loc[
        joined_df["building_id"].isna()
    ]
    .groupby("Type")["contract_ID"]
    .nunique()
    .reset_index(
        name="unmatched_unique_contracts"
    )
)

print(
    "Remaining unmatched UNIQUE contracts by Type:"
)
print(
    unmatched_by_type.to_string(index=False)
)

#######################################
# Check household-month uniqueness before filtering
#######################################

duplicate_household_months = joined_df.duplicated(
    subset=[
        "contract_ID",
        "month_year",
    ],
    keep=False,
)

print(
    "Rows involved in duplicate household-months "
    "after building assignment:",
    f"{duplicate_household_months.sum():,}"
)

if duplicate_household_months.any():
    duplicate_combinations = (
        joined_df.loc[
            duplicate_household_months,
            [
                "contract_ID",
                "month_year",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Distinct duplicated household-month combinations:",
        f"{duplicate_combinations:,}"
    )

    raise ValueError(
        "Building assignment created duplicate household-month rows."
    )

#######################################
# Keep only rows assigned to a building
#######################################

rows_before_building_filter = len(joined_df)

joined_df = joined_df.loc[
    joined_df["building_id"].notna()
].copy()

print(
    "Rows retained after requiring a building assignment:",
    f"{len(joined_df):,} / "
    f"{rows_before_building_filter:,} "
    f"({100 * len(joined_df) / rows_before_building_filter:.2f}%)"
)

#######################################
# Unique contracts matched by Type
#######################################

type_counts_merged = (
    joined_df
    .groupby("Type")["contract_ID"]
    .nunique()
    .reset_index(
        name="unique_contracts_merged"
    )
)

print(
    "Unique contracts by Type that merged with buildings:"
)

print(
    type_counts_merged.to_string(index=False)
)

#######################################
# Percent matched
#######################################

matched_unique_contracts = (
    joined_df["contract_ID"].nunique()
)

percent_matched = (
    100
    * matched_unique_contracts
    / total_unique_contracts_all
)

print(
    "\nPercent of ALL unique contracts matched "
    f"with a building: {percent_matched:.2f}%"
)

print(
    "Total unique contracts (ALL):",
    f"{total_unique_contracts_all:,}"
)

print(
    "Unique contracts matched:",
    f"{matched_unique_contracts:,}"
)

print("\nBuilding match types:")
print(
    joined_df["building_match_type"]
    .value_counts(dropna=False)
)

print("\nNearest-distance distribution:")
print(
    joined_df.loc[
        joined_df["building_match_type"].eq("nearest"),
        "dist_m",
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

#######################################
# Final checks
#######################################

multiple_buildings = (
    joined_df
    .groupby(
        [
            "contract_ID",
            "wkt",
        ]
    )["building_id"]
    .nunique()
    .gt(1)
    .sum()
)

print(
    "Contract-location combinations assigned "
    "to multiple buildings:",
    f"{multiple_buildings:,}"
)

if multiple_buildings > 0:
    raise ValueError(
        "Some contract-location combinations have "
        "multiple building IDs."
    )

final_household_month_duplicates = (
    joined_df.duplicated(
        subset=[
            "contract_ID",
            "month_year",
        ],
        keep=False,
    )
    .sum()
)

print(
    "Final duplicate household-month rows:",
    f"{final_household_month_duplicates:,}"
)

if final_household_month_duplicates > 0:
    raise ValueError(
        "Duplicate household-month rows remain."
    )

missing_keys = (
    location_lookup[["contract_ID", "wkt"]]
    .drop_duplicates()
    .merge(
        building_lookup[["contract_ID", "wkt"]]
        .drop_duplicates(),
        how="left",
        indicator=True,
    )
)

n_missing = (missing_keys["_merge"] == "left_only").sum()

print(f"Missing contract-location keys: {n_missing:,}")


lost_locations = (
    location_lookup[
        ["contract_ID", "wkt"]
    ]
    .merge(
        building_lookup[
            ["contract_ID", "wkt"]
        ],
        on=["contract_ID", "wkt"],
        how="left",
        indicator=True,
    )
    .loc[lambda x: x["_merge"] == "left_only"]
)

print(lost_locations)

#######################################
# Save
#######################################

print("\nUnique buildings assigned:")
print(joined_df["building_id"].nunique())

joined_df.to_parquet(
    output_path,
    engine="pyarrow",
    index=False,
)

print(
    "[Step 5] Saved merged data with building assignments to:"
)

print(output_path)

building_change_check = (
    joined_df[
        [
            "contract_ID",
            "month_year",
            "wkt",
            "building_id",
        ]
    ]
    .sort_values(
        ["contract_ID", "month_year"]
    )
    .copy()
)

building_change_check["previous_building_id"] = (
    building_change_check
    .groupby("contract_ID")["building_id"]
    .shift()
)

building_change_check["previous_wkt"] = (
    building_change_check
    .groupby("contract_ID")["wkt"]
    .shift()
)

building_change_check["building_changed"] = (
    building_change_check["building_id"]
    != building_change_check["previous_building_id"]
)

building_change_check["location_changed"] = (
    building_change_check["wkt"]
    != building_change_check["previous_wkt"]
)

unexpected_building_changes = building_change_check.loc[
    building_change_check["building_changed"]
    & ~building_change_check["location_changed"]
    & building_change_check["previous_building_id"].notna()
]

print(
    "Building changes without a WKT change:",
    f"{len(unexpected_building_changes):,}"
)