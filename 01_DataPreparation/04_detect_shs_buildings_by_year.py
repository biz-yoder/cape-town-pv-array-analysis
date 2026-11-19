# # Merges yearly SHS panel data (location + capacity) with buildings

# import pandas as pd
# import geopandas as gpd
# import os
# import time

# # Paths
# CSV_PATHS = {
#     # 2020: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023/2020/output_stage_4/prediction_merged_2023.csv",
#     2018: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2018/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2018_final.csv",
#     2019: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2019/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2019_final.csv",
#     2020: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2020/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2020_final.csv",
#     2021: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2021/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2021_final.csv",
#     2022: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2022/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2022_final.csv",
#     2023: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2023_final.csv"
# }

# BUILDINGS_PATH = "/home/ey53/vscode-server-backup/CapeTown_Workflow/capetown_buildings2.parquet"
# OUTPUT_DIR = "/home/ey53/vscode-server-backup/CapeTown_Workflow/4_out_05272026"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Set up
# CHUNK_SIZE = 50_000
# LOG_FILE = os.path.join(OUTPUT_DIR, "merge_log.txt")

# #######################################
# # Logging helper

# def log(msg):
#     ts = time.strftime("%Y-%m-%d %H:%M:%S")
#     full_msg = f"[{ts}] {msg}"
#     print(full_msg)
#     with open(LOG_FILE, "a") as f:
#         f.write(full_msg + "\n")

# #######################################
# # Clean col names

# def clean_columns(df):
#     """Ensure column names are safe for GeoPackage output."""
#     old_cols = df.columns.tolist()
#     new_cols = [
#         c.replace("[", "_").replace("]", "_").replace(",", "_").replace("/", "_")[:30]
#         for c in old_cols
#     ]
#     df.columns = new_cols
#     return df

# #######################################
# # Load buildings

# log("Loading building polygons...")
# start = time.time()
# buildings_gdf = gpd.read_parquet(BUILDINGS_PATH)
# if buildings_gdf.crs != "EPSG:4326":
#     buildings_gdf = buildings_gdf.to_crs("EPSG:4326")
# _ = buildings_gdf.sindex  # build spatial index
# log(f"Loaded {len(buildings_gdf):,} polygons in {time.time() - start:.1f}s")

# #######################################
# # MAIN

# for year, csv_path in CSV_PATHS.items():
#     log(f"\n--- Starting year {year} ---")

#     checkpoint_file = os.path.join(OUTPUT_DIR, f"{year}_done_chunks.txt")
#     done_chunks = set()
#     if os.path.exists(checkpoint_file):
#         with open(checkpoint_file) as f:
#             done_chunks = set(int(line.strip()) for line in f if line.strip().isdigit())
#         log(f"Resuming — {len(done_chunks)} chunks already completed.")

#     # Year-level counters
#     total_start_pv = total_dropped_area = total_dropped_nomatch = total_matched = 0

#     for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False), start=1):
#         if i in done_chunks:
#             log(f"Skipping chunk {i} (already done).")
#             continue

#         log(f"Processing chunk {i} ({len(chunk):,} rows)...")

#         # Filter only PV_normal
#         chunk = chunk[chunk["label"] == "PV_normal"]
#         start_pv = len(chunk)
#         total_start_pv += start_pv
#         log(f"Initial PV_normal rows: {start_pv:,}")

#         # Drop small area_m2 < 1.7
#         if "area_m2" in chunk.columns:
#             before_area = len(chunk)
#             chunk = chunk[chunk["area_m2"] >= 1.7]
#             dropped_area = before_area - len(chunk)
#             total_dropped_area += dropped_area
#             log(f"Dropped {dropped_area:,} rows (area_m2 < 1.7). Remaining: {len(chunk):,}")
#         else:
#             log("⚠️ No 'area_m2' column found — skipping area filter.")

#         if chunk.empty:
#             log(f"Chunk {i}: No PV_normal rows left after filtering, skipping.")
#             continue

#         expected_col = "polygon_centroid_GPS[lat,lon]"

#         if expected_col not in chunk.columns:
#             # Look for a similar column
#             possible_cols = [c for c in chunk.columns if c.startswith("polygon_centroid_GPS_lat_lon")]
#             if possible_cols:
#                 # Rename the first match
#                 chunk.rename(columns={possible_cols[0]: expected_col}, inplace=True)
#                 log(f"INFO: Renamed {possible_cols[0]} to {expected_col} for {year}, chunk {i}")
#             else:
#                 log(f"ERROR: Missing GPS column for {year}, chunk {i}")
#                 continue
        
#         # Robust GPS parsing
#         gps_col = "polygon_centroid_GPS[lat,lon]"

#         # Remove brackets/parentheses, strip spaces
#         coord_clean = chunk[gps_col].str.replace(r"[\(\)\[\]]", "", regex=True).str.strip()

#         # Split on comma into two columns
#         gps = coord_clean.str.split(",", expand=True)
#         chunk.loc[:, "lat"] = pd.to_numeric(gps[0].str.strip(), errors="coerce")
#         chunk.loc[:, "lon"] = pd.to_numeric(gps[1].str.strip(), errors="coerce")

#         # Drop rows with invalid coordinates
#         chunk = chunk.dropna(subset=["lat", "lon"])
#         if chunk.empty:
#             log(f"Chunk {i}: no valid coordinates, skipping.")
#             continue

#         # Convert to GeoDataFrame
#         gdf = gpd.GeoDataFrame(
#             chunk,
#             geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"]),
#             crs="EPSG:4326"
#         )

#         # Subset buildings for speed
#         minx, miny, maxx, maxy = gdf.total_bounds
#         b_subset = buildings_gdf.cx[minx:maxx, miny:maxy][['id', 'geometry']]
#         log(f"Spatial join subset: {len(b_subset):,} buildings")

#         # Spatial join
#         start_join = time.time()
#         merged = gpd.sjoin(gdf, b_subset, how="left", predicate="within")
#         join_time = time.time() - start_join
#         log(f"Join done in {join_time:.1f}s — {len(merged):,} rows")

#         merged['building_id'] = merged['index_right'].map(b_subset['id'])

#         print(f"Rows after spatial join: {len(merged):,}")
#         matched_rows = merged['building_id'].notna().sum()
#         total_rows = len(merged)
#         print(f"Matched rows: {matched_rows:,}/{total_rows:,}")

#         matched_rows = merged["building_id"].notna().sum()
#         total_rows = len(merged)
#         total_matched += matched_rows
#         match_rate = 100 * matched_rows / total_rows if total_rows else 0
#         unique_buildings = merged["building_id"].dropna().nunique()
#         log(f"Matched {matched_rows:,}/{total_rows:,} rows ({match_rate:.2f}%) to {unique_buildings:,} unique buildings")

#         # Compute change from baseline
#         dropped_no_match = start_pv - matched_rows
#         total_dropped_nomatch += dropped_no_match
#         log(f"Dropped {dropped_no_match:,} PV_normal rows (no building match, relative to start baseline).")

#         # Find unmatched PV
#         unmatched = merged[merged['building_id'].isna()].copy()
#         log(f"Unmatched PV points after sjoin: {len(unmatched):,}")

#         if not unmatched.empty:
#             # Drop existing 'index_right' to avoid conflict
#             if 'index_right' in unmatched.columns:
#                 unmatched = unmatched.drop(columns=['index_right'])

#             b_subset_free = b_subset[~b_subset.index.isin(merged["index_right"].dropna())].copy()
#             if not b_subset_free.empty:
#                 unmatched = unmatched.to_crs("EPSG:32734")
#                 b_subset_proj = b_subset_free.to_crs("EPSG:32734")
#                 b_subset_proj = b_subset_proj.reset_index().rename(columns={'index': 'building_index'})

#                 nearest = gpd.sjoin_nearest(
#                     unmatched,
#                     b_subset_proj[['building_index', 'geometry']],
#                     how='left',
#                     distance_col='dist_m'
#                 )

#                 max_distance = 15
#                 nearest_within = nearest[nearest['dist_m'] <= max_distance].copy()
#                 merged.loc[nearest_within.index, 'building_id'] = nearest_within['building_index'].map(b_subset_proj.set_index('building_index')['id'])

#                 # Compute distance summary
#                 dist_summary = nearest['dist_m'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
#                 log(f"Distance summary to nearest building (meters):\n{dist_summary}")
#                 log(f"Assigning only points within {max_distance:.1f} meters")

#                 # Keep only points within 99th percentile
#                 nearest_within = nearest[nearest['dist_m'] <= max_distance].copy()
#                 log(f"Number of points assigned via nearest building: {len(nearest_within):,}")

#                 # Update 'merged'
#                 merged.loc[nearest_within.index, 'index_right'] = nearest_within['building_index']

#                 # Continue processing
#                 merged["year"] = year
#                 merged = clean_columns(merged)

#         # Keep only rows matched to a building
#         merged_matched = merged[merged["building_id"].notna()].copy()

#         if not merged_matched.empty:
#             # Ensure area_m2 exists
#             if "area_m2" not in merged_matched.columns:
#                 log("⚠️ area_m2 missing — cannot aggregate.")
#             else:
#                 # Aggregate SHS per building
#                 building_sum = (
#                     merged_matched
#                     .groupby("building_id", as_index=False)
#                     .agg(
#                         total_shs_area_m2=("area_m2", "sum"),
#                         n_panels=("area_m2", "count"),
#                         prediction_ids=("prediction_id", list)
#                     )
#                 )

#                 building_sum["year"] = year

#                 # 🔥 Merge building geometry back in
#                 building_sum = building_sum.merge(
#                     buildings_gdf[['id', 'geometry']],
#                     left_on='building_id',
#                     right_on='id',
#                     how='left'
#                 ).drop(columns='id')

#                 # Convert to GeoDataFrame
#                 building_sum = gpd.GeoDataFrame(
#                     building_sum,
#                     geometry='geometry',
#                     crs=buildings_gdf.crs
#                 )

#                 building_sum = clean_columns(building_sum)
#                 print("building_sum columns:", building_sum.columns)
#                 merged = building_sum
#         else:
#             log("No matched buildings to aggregate.")
#             merged = pd.DataFrame(columns=["building_id", "total_shs_area_m2", "n_panels", "year", "geometry"])
            
#         # Save
#         chunk_parquet = os.path.join(OUTPUT_DIR, f"{year}_chunk{i}.parquet")
#         merged.to_parquet(chunk_parquet)
#         log(f"✅ Saved chunk {i} → {chunk_parquet}")

#         with open(checkpoint_file, "a") as f:
#             f.write(f"{i}\n")

#         del gdf, merged, b_subset

#     # Yearly summary
#     total_remaining = total_matched
#     log(f"\n=== YEAR {year} SUMMARY ===")
#     log(f"Total starting PV_normal rows: {total_start_pv:,}")
#     log(f"Total dropped (area < 1.7): {total_dropped_area:,}")
#     log(f"Total dropped (no building match): {total_dropped_nomatch:,}")
#     log(f"Total remaining matched: {total_remaining:,}")
#     # Avoid division by zero
#     retention_rate = (100 * total_remaining / total_start_pv) if total_start_pv != 0 else None

#     if retention_rate is None:
#         log("Retention rate: N/A (total_start_pv is zero)")
#     else:
#         log(f"Retention rate: {retention_rate:.2f}%")
#     log("=============================\n")

# log("✅ All years processed successfully.")


# Merges yearly SHS detection data with Overture building polygons.

from __future__ import annotations

import ast
import gc
import os
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


###############################################################################
# PATHS AND PARAMETERS
###############################################################################

CSV_PATHS = {
    2018: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2018/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2018_final.csv",
    2019: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2019/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2019_final.csv",
    2020: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2020/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2020_final.csv",
    2021: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2021/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2021_final.csv",
    2022: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2022/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2022_final.csv",
    2023: "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2023_final.csv",
}

BUILDINGS_PATH = "/home/ey53/vscode-server-backup/CapeTown_Workflow/capetown_buildings2.parquet"
OUTPUT_DIR = Path("/home/ey53/vscode-server-backup/CapeTown_Workflow/4_out_08042026")

CHUNK_SIZE = 50_000
MIN_PV_AREA_M2 = 1.7
MAX_NEAREST_DISTANCE_M = 15.0

GEOGRAPHIC_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32734"

LOG_FILE = OUTPUT_DIR / "merge_log.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


###############################################################################
# HELPERS
###############################################################################

def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(full_message + "\n")


def mark_chunk_complete(checkpoint_file: Path, chunk_number: int) -> None:
    with checkpoint_file.open("a", encoding="utf-8") as file:
        file.write(f"{chunk_number}\n")


def load_completed_chunks(checkpoint_file: Path) -> set[int]:
    if not checkpoint_file.exists():
        return set()

    completed: set[int] = set()
    with checkpoint_file.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if value.isdigit():
                completed.add(int(value))
    return completed


def find_gps_column(columns: Iterable[str]) -> str | None:
    expected = "polygon_centroid_GPS[lat,lon]"
    if expected in columns:
        return expected

    candidates = [
        column
        for column in columns
        if column.startswith("polygon_centroid_GPS_lat_lon")
        or column.startswith("polygon_centroid_GPS")
    ]
    return candidates[0] if candidates else None


def flatten_prediction_ids(values: pd.Series) -> list:
    output: list = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            output.extend(list(value))
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, list):
                        output.extend(parsed)
                        continue
                except (ValueError, SyntaxError):
                    pass
        output.append(value)
    return output


def safe_percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


###############################################################################
# LOAD BUILDINGS
###############################################################################

log("=" * 90)
log("LOADING BUILDING POLYGONS")
log("=" * 90)

building_start = time.time()
buildings_gdf = gpd.read_parquet(BUILDINGS_PATH)

if buildings_gdf.crs is None:
    raise ValueError("Building data have no CRS.")

if buildings_gdf.crs.to_string() != GEOGRAPHIC_CRS:
    log(f"Reprojecting buildings from {buildings_gdf.crs} to {GEOGRAPHIC_CRS}.")
    buildings_gdf = buildings_gdf.to_crs(GEOGRAPHIC_CRS)

required_building_columns = {"id", "geometry"}
missing_building_columns = required_building_columns.difference(buildings_gdf.columns)
if missing_building_columns:
    raise KeyError(f"Building file is missing required columns: {sorted(missing_building_columns)}")

before_valid_geometry = len(buildings_gdf)
buildings_gdf = buildings_gdf.loc[
    buildings_gdf.geometry.notna()
    & ~buildings_gdf.geometry.is_empty
    & buildings_gdf.geometry.is_valid
].copy()

log(f"Removed invalid/empty building geometries: {before_valid_geometry - len(buildings_gdf):,}")

buildings_metric_for_area = buildings_gdf[["id", "geometry"]].to_crs(METRIC_CRS)
building_area_lookup = pd.Series(
    buildings_metric_for_area.geometry.area.to_numpy(),
    index=buildings_metric_for_area["id"].to_numpy(),
    name="building_area_m2",
)
buildings_gdf["building_area_m2"] = buildings_gdf["id"].map(building_area_lookup)
_ = buildings_gdf.sindex

log(f"Loaded {len(buildings_gdf):,} valid building polygons in {time.time() - building_start:.1f} seconds.")
log(f"Unique building IDs: {buildings_gdf['id'].nunique():,}")

if buildings_gdf["id"].duplicated().any():
    duplicate_building_ids = buildings_gdf.loc[
        buildings_gdf["id"].duplicated(keep=False), "id"
    ].nunique()
    raise ValueError(f"Building file contains {duplicate_building_ids:,} duplicated building IDs.")


###############################################################################
# PROCESS EACH YEAR
###############################################################################

for year, csv_path in CSV_PATHS.items():
    year_start = time.time()

    log("")
    log("=" * 90)
    log(f"STARTING YEAR {year}")
    log("=" * 90)
    log(f"Input CSV: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    checkpoint_file = OUTPUT_DIR / f"{year}_done_chunks.txt"
    completed_chunks = load_completed_chunks(checkpoint_file)

    if completed_chunks:
        log(f"Resuming year {year}: {len(completed_chunks):,} chunks already completed.")
    else:
        log(f"Starting year {year} from the beginning.")

    total_csv_rows = 0
    total_start_pv = 0
    total_dropped_area = 0
    total_dropped_invalid_gps = 0
    total_direct_matched = 0
    total_nearest_matched = 0
    total_unmatched = 0
    total_final_matched = 0
    total_direct_overlap_detections = 0
    total_nearest_tie_detections = 0

    csv_reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk_number, chunk in enumerate(csv_reader, start=1):
        if chunk_number in completed_chunks:
            log(f"Skipping chunk {chunk_number} (checkpointed).")
            continue

        chunk_start_time = time.time()
        original_chunk_rows = len(chunk)
        total_csv_rows += original_chunk_rows

        log("")
        log("-" * 80)
        log(f"YEAR {year} | CHUNK {chunk_number} | RAW ROWS: {original_chunk_rows:,}")
        log("-" * 80)

        required_columns = {"label", "area_m2"}
        missing_required = required_columns.difference(chunk.columns)
        if missing_required:
            raise KeyError(
                f"Year {year}, chunk {chunk_number} is missing required columns: "
                f"{sorted(missing_required)}"
            )

        chunk = chunk.loc[chunk["label"].eq("PV_normal")].copy()
        start_pv = len(chunk)
        total_start_pv += start_pv
        log(f"PV_normal detections: {start_pv:,}")

        if chunk.empty:
            log("No PV_normal detections in this chunk. Marking complete.")
            mark_chunk_complete(checkpoint_file, chunk_number)
            continue

        chunk["area_m2"] = pd.to_numeric(chunk["area_m2"], errors="coerce")
        before_area = len(chunk)
        chunk = chunk.loc[
            chunk["area_m2"].notna()
            & chunk["area_m2"].ge(MIN_PV_AREA_M2)
        ].copy()

        dropped_area = before_area - len(chunk)
        total_dropped_area += dropped_area
        log(f"Dropped for missing/area < {MIN_PV_AREA_M2:.1f} m²: {dropped_area:,}")
        log(f"Remaining after area filter: {len(chunk):,}")

        if chunk.empty:
            log("No detections remain after area filtering. Marking complete.")
            mark_chunk_complete(checkpoint_file, chunk_number)
            continue

        gps_column = find_gps_column(chunk.columns)
        if gps_column is None:
            raise KeyError(
                f"Year {year}, chunk {chunk_number}: centroid GPS column was not found."
            )

        if gps_column != "polygon_centroid_GPS[lat,lon]":
            log(f"Using alternate GPS column: {gps_column}")

        coordinate_text = (
            chunk[gps_column]
            .astype("string")
            .str.replace(r"[\(\)\[\]]", "", regex=True)
            .str.strip()
        )
        coordinates = coordinate_text.str.split(",", n=1, expand=True)

        if coordinates.shape[1] < 2:
            raise ValueError(
                f"Year {year}, chunk {chunk_number}: GPS values could not be split."
            )

        chunk["lat"] = pd.to_numeric(coordinates[0].str.strip(), errors="coerce")
        chunk["lon"] = pd.to_numeric(coordinates[1].str.strip(), errors="coerce")

        before_gps = len(chunk)
        valid_coordinate_mask = (
            chunk["lat"].between(-90, 90)
            & chunk["lon"].between(-180, 180)
        )
        chunk = chunk.loc[valid_coordinate_mask].copy()

        dropped_invalid_gps = before_gps - len(chunk)
        total_dropped_invalid_gps += dropped_invalid_gps
        log(f"Dropped invalid coordinates: {dropped_invalid_gps:,}")
        log(f"Remaining geocoded detections: {len(chunk):,}")

        if chunk.empty:
            log("No valid coordinates remain. Marking complete.")
            mark_chunk_complete(checkpoint_file, chunk_number)
            continue

        chunk = chunk.reset_index(drop=True)
        chunk["pv_row_id"] = [
            f"{year}_{chunk_number}_{row_number}"
            for row_number in range(len(chunk))
        ]

        if "prediction_id" not in chunk.columns:
            log("WARNING: prediction_id is missing. Using pv_row_id instead.")
            chunk["prediction_id"] = chunk["pv_row_id"]

        duplicate_prediction_ids = chunk["prediction_id"].duplicated(keep=False).sum()
        log(f"Rows with duplicated prediction_id within this chunk: {duplicate_prediction_ids:,}")

        detection_gdf = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"]),
            crs=GEOGRAPHIC_CRS,
        )

        minx, miny, maxx, maxy = detection_gdf.total_bounds
        buildings_subset = (
            buildings_gdf.cx[minx:maxx, miny:maxy][
                ["id", "building_area_m2", "geometry"]
            ]
            .copy()
            .rename(columns={"id": "building_id"})
        )

        log(f"Buildings in chunk bounding box: {len(buildings_subset):,}")

        if buildings_subset.empty:
            direct_join = detection_gdf.copy()
            direct_join["building_id"] = pd.NA
            direct_join["building_area_m2"] = np.nan
            direct_join["building_match_type"] = "unmatched"
            direct_join["dist_m"] = np.nan
        else:
            direct_join_start = time.time()
            direct_join = gpd.sjoin(
                detection_gdf,
                buildings_subset,
                how="left",
                predicate="within",
            )
            log(
                "Direct within-building spatial join completed in "
                f"{time.time() - direct_join_start:.1f} seconds."
            )
            log(f"Rows returned by direct join: {len(direct_join):,}")

            direct_overlap_mask = direct_join.duplicated(
                subset=["pv_row_id"], keep=False
            )
            direct_overlap_rows = int(direct_overlap_mask.sum())
            direct_overlap_detections = int(
                direct_join.loc[direct_overlap_mask, "pv_row_id"].nunique()
            )
            total_direct_overlap_detections += direct_overlap_detections

            log(f"Rows involved in multiple containing-building matches: {direct_overlap_rows:,}")
            log(f"Distinct detections with multiple containing buildings: {direct_overlap_detections:,}")

            direct_join = (
                direct_join
                .sort_values(
                    ["pv_row_id", "building_area_m2", "building_id"],
                    na_position="last",
                )
                .drop_duplicates(subset=["pv_row_id"], keep="first")
                .copy()
            )

            if len(direct_join) != len(detection_gdf):
                raise ValueError("Direct join resolution did not return one row per PV detection.")

            direct_join["building_match_type"] = np.where(
                direct_join["building_id"].notna(), "within", "unmatched"
            )
            direct_join["dist_m"] = np.where(
                direct_join["building_id"].notna(), 0.0, np.nan
            )

        direct_matched = int(direct_join["building_id"].notna().sum())
        total_direct_matched += direct_matched
        log(
            f"Direct within-building matches: {direct_matched:,}/{len(direct_join):,} "
            f"({safe_percent(direct_matched, len(direct_join)):.2f}%)"
        )

        unmatched = direct_join.loc[direct_join["building_id"].isna()].copy()
        log(f"Detections requiring nearest-building search: {len(unmatched):,}")

        nearest_matched = 0

        if not unmatched.empty and not buildings_subset.empty:
            unmatched_for_nearest = unmatched[["pv_row_id", "geometry"]].copy()
            unmatched_metric = unmatched_for_nearest.to_crs(METRIC_CRS)

            buildings_metric = (
                buildings_subset[["building_id", "geometry"]]
                .drop(columns=["index_left", "index_right"], errors="ignore")
                .to_crs(METRIC_CRS)
            )

            nearest_join_start = time.time()
            nearest = gpd.sjoin_nearest(
                unmatched_metric,
                buildings_metric,
                how="left",
                distance_col="dist_m",
                max_distance=MAX_NEAREST_DISTANCE_M,
            )
            log(
                "Nearest-building spatial join completed in "
                f"{time.time() - nearest_join_start:.1f} seconds."
            )

            nearest_tie_mask = nearest.duplicated(subset=["pv_row_id"], keep=False)
            nearest_tie_rows = int(nearest_tie_mask.sum())
            nearest_tie_detections = int(
                nearest.loc[nearest_tie_mask, "pv_row_id"].nunique()
            )
            total_nearest_tie_detections += nearest_tie_detections

            log(f"Rows involved in nearest-building ties: {nearest_tie_rows:,}")
            log(f"Distinct detections with nearest-building ties: {nearest_tie_detections:,}")

            nearest = (
                nearest
                .sort_values(
                    ["pv_row_id", "dist_m", "building_id"],
                    na_position="last",
                )
                .drop_duplicates(subset=["pv_row_id"], keep="first")
                .copy()
            )

            valid_nearest = nearest.loc[
                nearest["building_id"].notna()
                & nearest["dist_m"].le(MAX_NEAREST_DISTANCE_M)
            ].copy()

            nearest_matched = int(valid_nearest["pv_row_id"].nunique())
            total_nearest_matched += nearest_matched

            if not valid_nearest.empty:
                nearest_building_map = valid_nearest.set_index("pv_row_id")["building_id"]
                nearest_distance_map = valid_nearest.set_index("pv_row_id")["dist_m"]

                nearest_mask = (
                    direct_join["building_id"].isna()
                    & direct_join["pv_row_id"].isin(nearest_building_map.index)
                )

                direct_join.loc[nearest_mask, "building_id"] = (
                    direct_join.loc[nearest_mask, "pv_row_id"].map(nearest_building_map)
                )
                direct_join.loc[nearest_mask, "dist_m"] = (
                    direct_join.loc[nearest_mask, "pv_row_id"].map(nearest_distance_map)
                )
                direct_join.loc[nearest_mask, "building_match_type"] = "nearest"

            distance_summary = valid_nearest["dist_m"].describe(
                percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
            )
            log(f"Nearest-match distance summary (meters):\n{distance_summary}")

        log(
            f"Nearest-building matches within {MAX_NEAREST_DISTANCE_M:.1f} m: "
            f"{nearest_matched:,}"
        )

        final_matched_rows = int(direct_join["building_id"].notna().sum())
        final_unmatched_rows = len(direct_join) - final_matched_rows
        total_final_matched += final_matched_rows
        total_unmatched += final_unmatched_rows

        log(
            f"Final matched detections: {final_matched_rows:,}/{len(direct_join):,} "
            f"({safe_percent(final_matched_rows, len(direct_join)):.2f}%)"
        )
        log(f"Final unmatched detections: {final_unmatched_rows:,}")
        log("Building match types:")
        log(direct_join["building_match_type"].value_counts(dropna=False).to_string())

        detection_duplicates = direct_join["pv_row_id"].duplicated(keep=False).sum()
        log(f"Duplicate PV-row IDs after all spatial matching: {detection_duplicates:,}")
        if detection_duplicates > 0:
            raise ValueError("Spatial matching produced duplicate PV detections.")

        matched_detections = direct_join.loc[
            direct_join["building_id"].notna()
        ].copy()

        if matched_detections.empty:
            chunk_summary = pd.DataFrame(
                columns=[
                    "building_id",
                    "total_shs_area_m2",
                    "n_panels",
                    "prediction_ids",
                    "year",
                ]
            )
        else:
            chunk_summary = (
                matched_detections
                .groupby("building_id", as_index=False)
                .agg(
                    total_shs_area_m2=("area_m2", "sum"),
                    n_panels=("area_m2", "count"),
                    prediction_ids=("prediction_id", list),
                )
            )
            chunk_summary["year"] = year

        duplicate_chunk_buildings = (
            chunk_summary["building_id"].duplicated(keep=False).sum()
            if not chunk_summary.empty
            else 0
        )
        log(f"Duplicate building IDs in chunk summary: {duplicate_chunk_buildings:,}")
        if duplicate_chunk_buildings > 0:
            raise ValueError("Chunk aggregation did not produce one row per building.")

        chunk_output = OUTPUT_DIR / f"{year}_chunk{chunk_number}.parquet"
        chunk_summary.to_parquet(chunk_output, index=False)

        log(f"Saved chunk summary: {chunk_output}")
        log(f"Buildings represented in chunk: {len(chunk_summary):,}")
        log(f"Chunk runtime: {time.time() - chunk_start_time:.1f} seconds")

        mark_chunk_complete(checkpoint_file, chunk_number)

        del chunk, detection_gdf, buildings_subset, direct_join, unmatched, matched_detections, chunk_summary
        gc.collect()

    ###########################################################################
    # CONSOLIDATE YEAR
    ###########################################################################

    log("")
    log("-" * 90)
    log(f"CONSOLIDATING YEAR {year}")
    log("-" * 90)

    year_chunk_files = sorted(OUTPUT_DIR.glob(f"{year}_chunk*.parquet"))
    log(f"Chunk files found for {year}: {len(year_chunk_files):,}")

    if not year_chunk_files:
        log(f"WARNING: No chunk outputs exist for {year}. Skipping yearly consolidation.")
        continue

    year_chunk_frames = []
    for chunk_file in year_chunk_files:
        frame = pd.read_parquet(chunk_file)
        if not frame.empty:
            year_chunk_frames.append(frame)

    if not year_chunk_frames:
        year_summary = gpd.GeoDataFrame(
            columns=[
                "building_id",
                "total_shs_area_m2",
                "n_panels",
                "prediction_ids",
                "year",
                "geometry",
            ],
            geometry="geometry",
            crs=GEOGRAPHIC_CRS,
        )
    else:
        year_combined = pd.concat(year_chunk_frames, ignore_index=True)

        log(f"Rows before yearly building aggregation: {len(year_combined):,}")
        log(f"Unique buildings before yearly aggregation: {year_combined['building_id'].nunique():,}")

        year_summary = (
            year_combined
            .groupby("building_id", as_index=False)
            .agg(
                total_shs_area_m2=("total_shs_area_m2", "sum"),
                n_panels=("n_panels", "sum"),
                prediction_ids=("prediction_ids", flatten_prediction_ids),
            )
        )
        year_summary["year"] = year

        log(f"Rows after yearly building aggregation: {len(year_summary):,}")

        if year_summary["building_id"].duplicated(keep=False).any():
            raise ValueError(f"Duplicate building IDs remain in the consolidated {year} output.")

        year_summary = year_summary.merge(
            buildings_gdf[["id", "geometry"]],
            left_on="building_id",
            right_on="id",
            how="left",
            validate="one_to_one",
        ).drop(columns="id")

        missing_year_geometry = year_summary["geometry"].isna().sum()
        log(f"Yearly building rows missing geometry: {missing_year_geometry:,}")
        if missing_year_geometry > 0:
            raise ValueError(
                f"{missing_year_geometry:,} building rows lack geometry in year {year}."
            )

        year_summary = gpd.GeoDataFrame(
            year_summary,
            geometry="geometry",
            crs=buildings_gdf.crs,
        )

    year_output = OUTPUT_DIR / f"shs_building_{year}.parquet"
    year_summary.to_parquet(year_output, index=False)
    log(f"Saved consolidated yearly output: {year_output}")

    aggregated_panel_count = (
        int(year_summary["n_panels"].sum()) if not year_summary.empty else 0
    )
    aggregated_area = (
        float(year_summary["total_shs_area_m2"].sum()) if not year_summary.empty else 0.0
    )

    log("")
    log(f"YEAR {year} SUMMARY")
    log(f"Total raw CSV rows read: {total_csv_rows:,}")
    log(f"Total PV_normal detections before filtering: {total_start_pv:,}")
    log(f"Total dropped for area/missing area: {total_dropped_area:,}")
    log(f"Total dropped for invalid GPS: {total_dropped_invalid_gps:,}")
    log(f"Total direct within-building matches: {total_direct_matched:,}")
    log(f"Total nearest-building matches: {total_nearest_matched:,}")
    log(f"Total final matched detections: {total_final_matched:,}")
    log(f"Total final unmatched detections: {total_unmatched:,}")
    log(
        "Total detections involved in overlapping direct matches: "
        f"{total_direct_overlap_detections:,}"
    )
    log(
        "Total detections involved in nearest ties: "
        f"{total_nearest_tie_detections:,}"
    )
    log(f"Unique buildings with SHS in {year}: {len(year_summary):,}")
    log(f"Aggregated panel detections in yearly output: {aggregated_panel_count:,}")
    log(f"Aggregated SHS area in yearly output: {aggregated_area:,.2f} m²")
    log(
        "Retention from post-area/post-GPS detections: "
        f"{safe_percent(total_final_matched, total_final_matched + total_unmatched):.2f}%"
    )

    if aggregated_panel_count != total_final_matched:
        raise ValueError(
            f"Year {year}: yearly n_panels sum ({aggregated_panel_count:,}) "
            f"does not equal matched detections ({total_final_matched:,})."
        )

    log(f"Year {year} completed in {time.time() - year_start:.1f} seconds.")
    log("=" * 90)

    del year_chunk_frames, year_summary
    if "year_combined" in locals():
        del year_combined
    gc.collect()


###############################################################################
# FINAL MULTI-YEAR QA
###############################################################################

log("")
log("=" * 90)
log("FINAL MULTI-YEAR QA")
log("=" * 90)

year_outputs = sorted(OUTPUT_DIR.glob("shs_building_*.parquet"))
log(f"Consolidated yearly files found: {len(year_outputs):,}")

if len(year_outputs) != len(CSV_PATHS):
    log(
        "WARNING: Number of yearly output files does not equal "
        "the number of configured years."
    )

multi_year_frames = []
for year_file in year_outputs:
    frame = pd.read_parquet(
        year_file,
        columns=[
            "building_id",
            "year",
            "total_shs_area_m2",
            "n_panels",
        ],
    )
    multi_year_frames.append(frame)

if multi_year_frames:
    multi_year = pd.concat(multi_year_frames, ignore_index=True)

    duplicate_building_years = multi_year.duplicated(
        subset=["building_id", "year"], keep=False
    ).sum()

    log(
        "Duplicate building-year rows across consolidated outputs: "
        f"{duplicate_building_years:,}"
    )

    if duplicate_building_years > 0:
        raise ValueError("Duplicate building-year rows remain.")

    yearly_qa = (
        multi_year
        .groupby("year")
        .agg(
            buildings=("building_id", "nunique"),
            panel_detections=("n_panels", "sum"),
            total_shs_area_m2=("total_shs_area_m2", "sum"),
        )
        .reset_index()
        .sort_values("year")
    )

    log(f"Final yearly QA table:\n{yearly_qa.to_string(index=False)}")

overall_unique_buildings = (
    multi_year["building_id"].nunique()
)

overall_building_years = len(multi_year)

overall_panel_detections = (
    multi_year["n_panels"].sum()
)

overall_shs_area_m2 = (
    multi_year["total_shs_area_m2"].sum()
)

log("")
log("OVERALL MULTI-YEAR SUMMARY")
log(
    "Unique buildings with detected SHS "
    f"across all years: {overall_unique_buildings:,}"
)
log(
    f"Total building-year observations: "
    f"{overall_building_years:,}"
)
log(
    f"Total panel detections across all years: "
    f"{overall_panel_detections:,}"
)
log(
    f"Total detected SHS area across all years: "
    f"{overall_shs_area_m2:,.2f} m²"
)

log("All configured years processed successfully.")

