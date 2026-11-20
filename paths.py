"""Centralized paths for the Cape Town rooftop solar analysis."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(
    os.environ.get("CAPETOWN_DATA_ROOT", REPO_ROOT / "data")
).expanduser().resolve()

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Final analysis datasets
PREPAID_PANEL = PROCESSED_DIR / "08_final_analysis_panels" / "prepaid_analysis_panel.parquet"
POSTPAID_PANEL = PROCESSED_DIR / "08_final_analysis_panels" / "postpaid_analysis_panel.parquet"


# Step 07: Monthly electricity consumption and load shedding
STEP7_PANEL = (
    PROCESSED_DIR
    / "07_monthly_load_shedding"
    / "monthly_consumption_with_outages.parquet"
)


# Analysis output directories
REGRESSION_DIR = OUTPUT_DIR / "regressions"
EVENT_STUDY_DIR = OUTPUT_DIR / "event_studies"
TIMING_SENSITIVITY_DIR = OUTPUT_DIR / "timing_sensitivity"
REVENUE_DIR = OUTPUT_DIR / "revenue"

# Historical registration input used by the revenue analysis
REVENUE_REGISTRATION_PANEL = RAW_DIR / "revenue_registration_august.parquet"

SHS_CLASSIFIED_PANEL = PROCESSED_DIR / "05_shs_classification" / "combined.parquet"
SHS_REGISTRATION_PANEL = PROCESSED_DIR / "05_shs_registration" / "with_sseg_reg.parquet"
LOAD_SHEDDING_BLOCK_PANEL = PROCESSED_DIR / "06_load_shedding_blocks" / "merged_with_blocks.parquet"
REGISTRATION_RECORDS = RAW_DIR / "registered.parquet"
LOAD_SHEDDING_BLOCKS = RAW_DIR / "Load_shedding_Blocks.geojson"
LOAD_SHEDDING_SCHEDULE = RAW_DIR / "Loadshedding_schedule.csv"
RAW_PREPAID_DIR = RAW_DIR / "Prepaid_parquet"

RAW_POSTPAID_PANEL = RAW_DIR / "postpaid_2018_2025.parquet"

RAW_CONTRACT_LOCATIONS = RAW_DIR / "ContractLocations"

RAW_BUILDINGS = RAW_DIR / "capetown_buildings2.parquet"

PV_DETECTION_DIR = RAW_DIR / "pv_detections"

MONTHLY_DIR = PROCESSED_DIR / "01_monthly_consumption"

PREPAID_MONTHLY = MONTHLY_DIR / "final_monthly_prepaid.parquet"

POSTPAID_MONTHLY = MONTHLY_DIR / "final_monthly_postpaid.parquet"

CONTRACT_LOCATIONS = PROCESSED_DIR / "02_contract_locations" / "new_location_total.parquet"

HISTORICAL_CONTRACT_LOCATIONS = RAW_DIR / "historical_contract_locations.parquet"

CONTRACT_LOCATION_MERGE = PROCESSED_DIR / "02_contract_location_merge" / "out_contract_with_location.parquet"

HISTORICAL_CONTRACT_LOCATION_MERGE = RAW_DIR / "historical_contract_location_merge.parquet"

CONTRACT_BUILDINGS = PROCESSED_DIR / "03_contract_buildings" / "out_contractlocation_with_building.parquet"

SHS_BUILDINGS_DIR = PROCESSED_DIR / "04_shs_buildings"

SHS_MATCHED_DIR = PROCESSED_DIR / "05_shs_matched"
