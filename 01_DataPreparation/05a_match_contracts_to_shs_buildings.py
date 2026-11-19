# Match contract-month observations to yearly SHS building detections.

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd


###############################################################################
# PATHS AND PARAMETERS
###############################################################################

BUILD_SHS_DIR = Path(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "4_out_08042026"
)

CONTRACT_BUILD_FILE = Path(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "3_out_08032026/out_contractlocation_with_building.parquet"
)

OUTPUT_DIR = Path(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "5a_out_08042026"
)

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


###############################################################################
# HELPERS
###############################################################################

def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_prediction_ids(value):
    """Return prediction IDs as a Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
        except Exception:
            pass
    return []


###############################################################################
# LOAD CONTRACT-BUILDING PANEL
###############################################################################

log("=" * 90)
log("LOADING CONTRACT-BUILDING PANEL")
log("=" * 90)

if not CONTRACT_BUILD_FILE.exists():
    raise FileNotFoundError(
        f"Contract-building file does not exist: {CONTRACT_BUILD_FILE}"
    )

load_start = time.time()

con = duckdb.connect()
contract_build = con.execute(
    f"""
    SELECT *
    FROM read_parquet('{CONTRACT_BUILD_FILE}')
    """
).fetchdf()
con.close()

log(
    f"Loaded {len(contract_build):,} contract-month rows "
    f"in {time.time() - load_start:.1f} seconds."
)
log(f"Unique contracts: {contract_build['contract_ID'].nunique():,}")
log(f"Columns loaded: {len(contract_build.columns):,}")

if "building_id" not in contract_build.columns:
    if "id" in contract_build.columns:
        contract_build = contract_build.rename(columns={"id": "building_id"})
        log("Renamed id to building_id.")
    else:
        raise KeyError(
            "Contract-building data contain neither 'building_id' nor 'id'."
        )

contract_build["month_year"] = pd.to_datetime(
    contract_build["month_year"],
    errors="coerce",
)

invalid_dates = contract_build["month_year"].isna().sum()
log(f"Rows with invalid month_year: {invalid_dates:,}")

if invalid_dates > 0:
    raise ValueError("Invalid month_year values remain in the contract panel.")

contract_build["year"] = contract_build["month_year"].dt.year
contract_build["month"] = contract_build["month_year"].dt.month

missing_building_ids = contract_build["building_id"].isna().sum()
log(f"Rows missing building_id: {missing_building_ids:,}")

input_duplicate_rows = contract_build.duplicated(
    subset=["contract_ID", "month_year"],
    keep=False,
).sum()

input_duplicate_combinations = (
    contract_build.loc[
        contract_build.duplicated(
            subset=["contract_ID", "month_year"],
            keep=False,
        ),
        ["contract_ID", "month_year"],
    ]
    .drop_duplicates()
    .shape[0]
)

log(
    "Input rows involved in duplicate contract-months: "
    f"{input_duplicate_rows:,}"
)
log(
    "Distinct duplicated input contract-month combinations: "
    f"{input_duplicate_combinations:,}"
)

if input_duplicate_rows > 0:
    raise ValueError(
        "The input contract-building panel is not unique by "
        "contract_ID and month_year."
    )


###############################################################################
# PROCESS EACH IMAGERY YEAR
###############################################################################

year_summary_rows = []

for year in YEARS:
    year_start = time.time()

    log("")
    log("=" * 90)
    log(f"PROCESSING YEAR {year}")
    log("=" * 90)

    output_file = OUTPUT_DIR / f"merged_contract_SHS_{year}.parquet"
    shs_file = BUILD_SHS_DIR / f"shs_building_{year}.parquet"

    log(f"SHS input: {shs_file}")
    log(f"Output file: {output_file}")

    if not shs_file.exists():
        raise FileNotFoundError(
            f"Consolidated SHS file does not exist for {year}: {shs_file}"
        )

    contracts_year = contract_build.loc[
        contract_build["year"].eq(year)
    ].copy()

    rows_before_merge = len(contracts_year)
    contracts_before_merge = contracts_year["contract_ID"].nunique()

    log(f"Contract-month rows in {year}: {rows_before_merge:,}")
    log(f"Unique contracts in {year}: {contracts_before_merge:,}")

    if contracts_year.empty:
        log(f"No contract observations exist for {year}; skipping.")
        continue

    shs_gdf = gpd.read_parquet(shs_file)

    required_shs_columns = {
        "building_id",
        "total_shs_area_m2",
        "n_panels",
        "prediction_ids",
    }

    missing_shs_columns = required_shs_columns.difference(shs_gdf.columns)

    if missing_shs_columns:
        raise KeyError(
            f"SHS file for {year} is missing required columns: "
            f"{sorted(missing_shs_columns)}"
        )

    shs_gdf = shs_gdf.loc[
        shs_gdf["building_id"].notna()
    ].copy()

    log(f"SHS building rows loaded: {len(shs_gdf):,}")
    log(f"Unique SHS buildings: {shs_gdf['building_id'].nunique():,}")

    duplicate_shs_rows = shs_gdf["building_id"].duplicated(
        keep=False
    ).sum()

    duplicate_shs_buildings = (
        shs_gdf.loc[
            shs_gdf["building_id"].duplicated(keep=False),
            "building_id",
        ]
        .nunique()
    )

    log(f"Rows involved in duplicate SHS building IDs: {duplicate_shs_rows:,}")
    log(f"Distinct duplicated SHS building IDs: {duplicate_shs_buildings:,}")

    if duplicate_shs_rows > 0:
        raise ValueError(
            f"The consolidated SHS file for {year} is not unique by building_id."
        )

    shs_lookup = shs_gdf[
        [
            "building_id",
            "total_shs_area_m2",
            "n_panels",
            "prediction_ids",
        ]
    ].copy()

    shs_lookup["total_shs_area_m2"] = pd.to_numeric(
        shs_lookup["total_shs_area_m2"],
        errors="coerce",
    )

    shs_lookup["n_panels"] = pd.to_numeric(
        shs_lookup["n_panels"],
        errors="coerce",
    )

    invalid_area = shs_lookup["total_shs_area_m2"].isna().sum()
    invalid_panel_count = shs_lookup["n_panels"].isna().sum()

    log(f"SHS buildings with invalid area: {invalid_area:,}")
    log(f"SHS buildings with invalid n_panels: {invalid_panel_count:,}")

    if invalid_area > 0 or invalid_panel_count > 0:
        raise ValueError(
            f"Invalid SHS area or panel counts found in the {year} SHS lookup."
        )

    shs_lookup["prediction_ids"] = (
        shs_lookup["prediction_ids"].apply(normalize_prediction_ids)
    )

    contracts_per_building = (
        contracts_year.loc[
            contracts_year["building_id"].notna(),
            ["building_id", "contract_ID"],
        ]
        .drop_duplicates()
        .groupby("building_id")["contract_ID"]
        .nunique()
    )

    log("Contracts per building distribution:")
    log(
        contracts_per_building.describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
        ).to_string()
    )

    shs_contract_counts = contracts_per_building.loc[
        contracts_per_building.index.isin(shs_lookup["building_id"])
    ]

    shs_buildings_multiple_contracts = shs_contract_counts.gt(1).sum()
    shs_buildings_total_linked = len(shs_contract_counts)

    log(
        "SHS buildings linked to more than one contract: "
        f"{shs_buildings_multiple_contracts:,}/"
        f"{shs_buildings_total_linked:,}"
    )

    if not shs_contract_counts.empty:
        log(
            "Maximum unique contracts linked to one SHS building: "
            f"{int(shs_contract_counts.max()):,}"
        )

    merge_start = time.time()

    contracts_merged = contracts_year.merge(
        shs_lookup,
        on="building_id",
        how="left",
        validate="many_to_one",
    )

    log(
        "Direct building-ID merge completed in "
        f"{time.time() - merge_start:.1f} seconds."
    )

    rows_after_merge = len(contracts_merged)

    log(f"Rows before SHS merge: {rows_before_merge:,}")
    log(f"Rows after SHS merge: {rows_after_merge:,}")

    if rows_after_merge != rows_before_merge:
        raise ValueError(
            f"SHS merge changed the number of rows for {year}: "
            f"{rows_before_merge:,} -> {rows_after_merge:,}."
        )

    ###############################################################################
    # Allocate building SHS equally across contracts in each building-month
    ###############################################################################

    contracts_per_building_month = (
        contracts_merged
        .groupby(["building_id", "month_year"])["contract_ID"]
        .nunique()
        .rename("n_contracts_building_month")
        .reset_index()
    )

    contracts_merged = contracts_merged.merge(
        contracts_per_building_month,
        on=["building_id", "month_year"],
        how="left",
        validate="many_to_one",
    )

    invalid_contract_counts = (
        contracts_merged["n_contracts_building_month"]
        .isna()
        | contracts_merged["n_contracts_building_month"].le(0)
    )

    log(
        "Rows with missing or nonpositive building-month "
        f"contract counts: {invalid_contract_counts.sum():,}"
    )

    if invalid_contract_counts.any():
        raise ValueError(
            "Some rows have missing or nonpositive "
            "n_contracts_building_month."
        )
    
    log("Unique building-month contract-count distribution:")
    log(
        contracts_per_building_month[
            "n_contracts_building_month"
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
        .to_string()
    )

    contracts_merged["shs_label_numeric"] = (
        contracts_merged["total_shs_area_m2"]
        .notna()
        .astype("int8")
    )

    # contracts_merged["SHS_area_m2"] = (
    #     contracts_merged["total_shs_area_m2"]
    #     .fillna(0.0)
    #     /
    #     contracts_merged["n_contracts_building_month"]
    # )
    contracts_merged["building_SHS_area_m2"] = (
        contracts_merged["total_shs_area_m2"]
        .fillna(0.0)
    )

    contracts_merged["SHS_area_per_contract"] = (
        contracts_merged["building_SHS_area_m2"]
        /
        contracts_merged["n_contracts_building_month"]
    )

    # Compatibility aliases for downstream scripts.
    contracts_merged["SHS_area_m2"] = (
        contracts_merged["SHS_area_per_contract"]
    )

    contracts_merged["shs_area_per_contract"] = (
        contracts_merged["SHS_area_per_contract"]
    )

    # total_before = contracts_merged["total_shs_area_m2"].fillna(0).sum()

    # total_after = (
    #     contracts_merged["SHS_area_m2"]
    #     * contracts_merged["n_contracts_building_month"]
    # ).sum()

    # log(f"Total SHS area before allocation: {total_before:,.2f}")
    # log(f"Total SHS area after allocation:  {total_after:,.2f}")

    # Check that allocated SHS area sums back to the building total
    # within each building-month.
    allocation_check = (
        contracts_merged.loc[
            contracts_merged["shs_label_numeric"].eq(1)
        ]
        .groupby(
            ["building_id", "month_year"],
            as_index=False,
        )
        .agg(
            building_shs_area=(
                "total_shs_area_m2",
                "first",
            ),
            allocated_shs_area=(
                "SHS_area_m2",
                "sum",
            ),
            n_contracts=(
                "contract_ID",
                "nunique",
            ),
        )
    )

    allocation_check["difference"] = (
        allocation_check["allocated_shs_area"]
        - allocation_check["building_shs_area"]
    )

    if allocation_check.empty:
        max_allocation_difference = 0.0
    else:
        max_allocation_difference = (
            allocation_check["difference"]
            .abs()
            .max()
        )

    log(
        "Maximum absolute building-month allocation difference: "
        f"{max_allocation_difference:.10f} m²"
    )

    if max_allocation_difference > 1e-8:
        raise ValueError(
            "Allocated contract-level SHS area does not sum back "
            "to building-level SHS area."
        )

    contracts_merged["n_panels"] = (
        contracts_merged["n_panels"]
        .fillna(0)
        .astype(int)
    )

    contracts_merged["prediction_ids"] = (
        contracts_merged["prediction_ids"]
        .apply(normalize_prediction_ids)
    )

    # # Compatibility alias for older scripts.
    # contracts_merged["shs_area_per_contract"] = (
    #     contracts_merged["SHS_area_m2"]
    # )

    positive_contract_months = int(
        contracts_merged["shs_label_numeric"].sum()
    )

    positive_unique_contracts = (
        contracts_merged.loc[
            contracts_merged["shs_label_numeric"].eq(1),
            "contract_ID",
        ]
        .nunique()
    )

    positive_unique_buildings = (
        contracts_merged.loc[
            contracts_merged["shs_label_numeric"].eq(1),
            "building_id",
        ]
        .nunique()
    )

    log(
        f"SHS-positive contract-month rows: "
        f"{positive_contract_months:,}/{rows_after_merge:,} "
        f"({100 * positive_contract_months / rows_after_merge:.2f}%)"
    )

    log(
        f"Unique contracts linked to an SHS building: "
        f"{positive_unique_contracts:,}/{contracts_before_merge:,} "
        f"({100 * positive_unique_contracts / contracts_before_merge:.2f}%)"
    )

    log(
        f"Unique SHS buildings represented in contract panel: "
        f"{positive_unique_buildings:,}"
    )

    unlinked_shs_buildings = (
        shs_lookup.loc[
            ~shs_lookup["building_id"].isin(
                contracts_year["building_id"]
            ),
            "building_id",
        ]
        .nunique()
    )

    log(
        f"SHS buildings with no linked contract in {year}: "
        f"{unlinked_shs_buildings:,}"
    )

    duplicate_contract_month_rows = (
        contracts_merged.duplicated(
            subset=["contract_ID", "month_year"],
            keep=False,
        )
        .sum()
    )

    duplicate_contract_month_combinations = (
        contracts_merged.loc[
            contracts_merged.duplicated(
                subset=["contract_ID", "month_year"],
                keep=False,
            ),
            ["contract_ID", "month_year"],
        ]
        .drop_duplicates()
        .shape[0]
    )

    log(
        "Rows involved in duplicate contract-months after SHS merge: "
        f"{duplicate_contract_month_rows:,}"
    )

    log(
        "Distinct duplicated contract-month combinations after SHS merge: "
        f"{duplicate_contract_month_combinations:,}"
    )

    if duplicate_contract_month_rows > 0:
        raise ValueError(
            f"The SHS merge created duplicate contract-month observations "
            f"in {year}."
        )

    building_total_area_variation = (
        contracts_merged.loc[
            contracts_merged["shs_label_numeric"].eq(1)
        ]
        .groupby("building_id")["total_shs_area_m2"]
        .nunique()
    )

    buildings_with_inconsistent_total_area = (
        building_total_area_variation.gt(1).sum()
    )

    log(
        "SHS buildings with inconsistent total building area "
        f"within the same year: "
        f"{buildings_with_inconsistent_total_area:,}"
    )

    if buildings_with_inconsistent_total_area > 0:
        raise ValueError(
            f"Some buildings have inconsistent total SHS area "
            f"within {year}."
        )

    if "geometry" in contracts_merged.columns:
        contracts_merged["geometry_wkt"] = (
            contracts_merged["geometry"]
            .apply(
                lambda geometry: (
                    geometry.wkt
                    if geometry is not None
                    and not pd.isna(geometry)
                    else None
                )
            )
        )

    contracts_merged.to_parquet(
        output_file,
        engine="pyarrow",
        index=False,
    )

    log(f"Saved {year} output: {output_file}")
    log(f"Saved rows: {len(contracts_merged):,}")
    log(
        f"Year {year} completed in "
        f"{time.time() - year_start:.1f} seconds."
    )

    year_summary_rows.append(
        {
            "year": year,
            "contract_month_rows": len(contracts_merged),
            "unique_contracts": contracts_before_merge,
            "shs_positive_contract_months": positive_contract_months,
            "shs_positive_unique_contracts": positive_unique_contracts,
            "shs_buildings_in_contract_panel": positive_unique_buildings,
            "shs_buildings_without_contract": unlinked_shs_buildings,
            "duplicate_contract_month_rows": duplicate_contract_month_rows,
        }
    )


###############################################################################
# FINAL QA SUMMARY
###############################################################################

log("")
log("=" * 90)
log("FINAL YEARLY QA SUMMARY")
log("=" * 90)

year_summary = pd.DataFrame(year_summary_rows)

if not year_summary.empty:
    log("\n" + year_summary.to_string(index=False))

    summary_output = OUTPUT_DIR / "contract_shs_yearly_QA.csv"

    year_summary.to_csv(
        summary_output,
        index=False,
    )

    log(f"Saved yearly QA summary: {summary_output}")

log("All imagery years processed successfully.")