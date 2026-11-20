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
