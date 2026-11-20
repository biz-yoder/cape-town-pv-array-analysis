# Cape Town Rooftop Solar and Electricity Consumption
## Data Preparation Pipeline

### Overview

This directory contains the Python scripts used to
construct the household-month electricity consumption panel for the
Cape Town rooftop solar analysis.

The pipeline combines:
- Monthly prepaid and postpaid electricity consumption records.
- Electricity contract and geographic location information.
- Building footprint data.
- Rooftop solar (SHS/PV) detections derived from aerial imagery.
- Official SSEG registration records.
- Load-shedding block assignments and monthly load-shedding exposure.

The final outputs are separate prepaid and postpaid household-month
panels used in the subsequent econometric analysis.

Scripts are numbered in processing order. Letter suffixes indicate
multiple operations within a processing stage.

---

## Current Path Configuration

The current scripts use centralized paths defined in the repository-root
`paths.py`. The paths below describe the current code configuration. Required inputs are not distributed with this repository.

Set the optional environment variable:

    export CAPETOWN_DATA_ROOT=/path/to/private/capetown-data

By default, `CAPETOWN_DATA_ROOT` is the repository's `data/` directory.
The following paths are relative to that data root.

| Step | Current output |
|---|---|
| 01a | `processed/01_monthly_consumption/final_monthly_prepaid.parquet` |
| 01b | `processed/01_monthly_consumption/final_monthly_postpaid.parquet` |
| 02a | `processed/02_contract_locations/new_location_total.parquet` |
| 02b | `processed/02_contract_location_merge/out_contract_with_location.parquet` |
| 03 | `processed/03_contract_buildings/out_contractlocation_with_building.parquet` |
| 04 | `processed/04_shs_buildings/shs_building_{year}.parquet` |
| 05a | `processed/05_shs_matched/merged_contract_SHS_{year}.parquet` |
| 05b | `processed/05_shs_classification/combined.parquet` |
| 05c | `processed/05_shs_registration/with_sseg_reg.parquet` |
| 06 | `processed/06_load_shedding_blocks/merged_with_blocks.parquet` |
| 07a | `processed/07_monthly_load_shedding/monthly_consumption_with_outages.parquet` |
| 08 | `processed/08_final_analysis_panels/prepaid_analysis_panel.parquet` and `postpaid_analysis_panel.parquet` |

External inputs belong under `raw/`, including prepaid transactions,
postpaid consumption, contract locations, building footprints,
registration records, and load-shedding data.

Annual PV detection files are expected at
`raw/pv_detections/{year}/prediction_merged_{year}_final.csv`
for 2018 through 2023.

Two historical intermediate datasets are required:

- `raw/historical_contract_locations.parquet` (Step 02b).
- `raw/historical_contract_location_merge.parquet` (Step 03).

These are intentionally distinct from the outputs of Steps 02a and
02b, respectively. Their numerical equivalence has not been verified.

The historical QA statistics below describe original execution runs.
The complete standardized pipeline has not been rerun.

---

## 1. Pipeline Structure

### Step 1: Construct Monthly Electricity Consumption Panels

**01a_create_monthly_prepaid.py**

Processes prepaid electricity transaction Parquet files from
`raw/Prepaid_parquet/` and aggregates transactions into monthly
household-level observations.

Output:
- `processed/01_monthly_consumption/final_monthly_prepaid.parquet`

**01b_create_monthly_postpaid.py**

Processes the postpaid transaction panel from
`raw/postpaid_2018_2025.parquet`.

Transactions are aggregated by contract and month.

Output:
- `processed/01_monthly_consumption/final_monthly_postpaid.parquet`

Verification:
- 6,714,889 rows.
- Five columns: contract identifier, tariff type, month,
  electricity consumption, and month-year.

### Step 2: Import and Merge Contract Locations

**02a_import_contract_locations.py**

Imports electricity contract location CSV files from
`raw/ContractLocations/`.

Records are filtered to `business_area == "TSES"`.

The resulting dataset retains contract identifiers, move-in
and move-out dates, activity status, geographic coordinates
(WKT), ward, suburb, electricity region, and related fields.

Output:
- `processed/02_contract_locations/new_location_total.parquet`

Dataset verification:
- 1,574,644 rows.
- 12 columns matching the fields selected in the script.

The Step 02b script uses a separate historical contract-location
dataset rather than the current Step 02a output. Their numerical
equivalence has not been verified.

**02b_merge_contracts_with_locations.py**

Combines prepaid and postpaid monthly consumption records
with contract location information.

Inputs:
- `processed/01_monthly_consumption/final_monthly_prepaid.parquet`
- `processed/01_monthly_consumption/final_monthly_postpaid.parquet`
- `raw/historical_contract_locations.parquet`

Output:
- `processed/02_contract_location_merge/out_contract_with_location.parquet`

Reported QA:
- 41,179,907 final rows.
- 812,025 unique contracts.
- No remaining duplicate household-month observations.
- 97.2% of rows have location information.

The historical intermediate dataset consumed by Step 03 has not been verified as numerically identical to the current Step 02b output.

### Step 3: Assign Contracts to Building Footprints

**03_assign_contracts_to_buildings.py**

Spatially matches contract locations to building footprints
using the contract-location dataset and building polygons.

Inputs:
- `raw/historical_contract_location_merge.parquet`
- `raw/capetown_buildings2.parquet`

Output:
- `processed/03_contract_buildings/out_contractlocation_with_building.parquet`

Reported QA:
- 812,025 unique contracts.
- 771,017 contracts matched to a building.
- 94.95% of unique contracts matched.
- 366,601 unique buildings assigned.
- Median nearest-building distance: 1.74 meters.
- No duplicate household-month observations.
- No contract-location combinations assigned to multiple buildings.
- No building changes without a corresponding location change.

### Step 4: Detect Rooftop SHS by Building and Year

**04_detect_shs_buildings_by_year.py**

Matches annual Mask2Former rooftop PV detections to
building footprints.

Inputs:
- Annual Mask2Former prediction CSV files at `raw/pv_detections/{year}/prediction_merged_{year}_final.csv` for 2018–2023.
- `raw/capetown_buildings2.parquet`

Processing parameters:
- Chunk size: 50,000.
- Minimum PV area: 1.7 m².
- Maximum nearest-building distance: 15 meters.
- Geographic CRS: EPSG:4326.
- Metric CRS: EPSG:32734.

Outputs:
- `processed/04_shs_buildings/shs_building_2018.parquet`
- `processed/04_shs_buildings/shs_building_2019.parquet`
- `processed/04_shs_buildings/shs_building_2020.parquet`
- `processed/04_shs_buildings/shs_building_2021.parquet`
- `processed/04_shs_buildings/shs_building_2022.parquet`
- `processed/04_shs_buildings/shs_building_2023.parquet`

Reported multi-year QA:
- 40,072 unique buildings with detected SHS.
- 88,491 building-year observations.
- 185,959 panel detections.
- 4,000,067.33 m² total detected SHS area.
- Zero duplicate building-year rows.
- All six imagery years processed successfully.

### Step 5: Construct SHS Treatment Variables

**05a_match_contracts_to_shs_buildings.py**

Matches contract-building observations to annual rooftop
SHS building detections.

Inputs:
- `processed/03_contract_buildings/out_contractlocation_with_building.parquet`
- Annual SHS building detections from Step 4.

Outputs:
- Annual contract-SHS merged datasets in `processed/05_shs_matched/`.

Reported QA:
- No duplicate contract-month observations.
- No inconsistent building areas within the same year.
- Six imagery years processed successfully.

**05b_apply_shs_classification_assumptions.py**

Combines annual contract-SHS datasets and applies the
constructed SHS classification and area assumptions.

Input:
- Annual contract-SHS datasets from Step 5a.

Output:
- `processed/05_shs_classification/combined.parquet`

**05c_merge_shs_registration_data.py**

Combines the contract-month panel with official SSEG
registration information.

Inputs:
- `processed/05_shs_classification/combined.parquet`
- `raw/registered.parquet`

Output:
- `processed/05_shs_registration/with_sseg_reg.parquet`

Reported QA:
- No duplicate account-month observations.
- SHS classification consistency verified across years.
- Registered capacity and detected SHS capacity retained
  as separate variables.

### Step 6: Assign Load-Shedding Blocks

**06_assign_load_shedding_blocks.py**

Assigns geographic load-shedding blocks to contract
locations using the load-shedding block boundaries.

Inputs:
- `processed/05_shs_registration/with_sseg_reg.parquet`
- `raw/Load_shedding_Blocks.geojson`

Output:
- `processed/06_load_shedding_blocks/merged_with_blocks.parquet`

Reported QA:
- 39,042,040 rows.
- 767,325 unique accounts with assigned blocks.
- 2,786 unique accounts without assigned blocks.
- Zero accounts assigned to multiple blocks over time.
- Zero account-location pairs assigned to multiple blocks.

### Step 7: Add Monthly Load-Shedding Exposure

**07a_calculate_monthly_load_shedding.py**

Calculates monthly load-shedding duration measures and
merges them into the household-month electricity panel.

Inputs:
- `processed/06_load_shedding_blocks/merged_with_blocks.parquet`
- `raw/Loadshedding_schedule.csv`

Output:
- `processed/07_monthly_load_shedding/monthly_consumption_with_outages.parquet`

Reported QA:
- 39,042,040 rows.
- 770,111 unique accounts before and after the merge.
- Zero duplicate account-month observations.
- Zero block-months with inconsistent duration values.
- 132,045 rows with missing total load-shedding duration.

### Step 8: Construct Final Analysis Panels

**08_build_final_analysis_panels.py**

Formats the final prepaid and postpaid household-month
datasets for the econometric analysis.

Applies a building-level restriction, retaining observations
associated with buildings having no more than five contracts:

`max_contracts_building_panel <= 5`.

Inputs:
- `processed/07_monthly_load_shedding/monthly_consumption_with_outages.parquet`

Outputs:
- `processed/08_final_analysis_panels/prepaid_analysis_panel.parquet`
- `processed/08_final_analysis_panels/postpaid_analysis_panel.parquet`

Reported QA:
- Prepaid: 33,967,252 rows.
- Postpaid: 5,074,788 rows.
- Prepaid unique accounts: 661,412.
- Postpaid unique accounts: 108,707.
- 172,744 PV household-month observations removed from
  236 buildings exceeding the five-contract threshold.
- Zero PV observations remaining in buildings with
  more than five contracts in the final panels.

---

## 2. Final Analysis Dataset

The final formatted datasets are used in the subsequent
econometric analysis.

Prepaid and postpaid data are retained separately.

The final analysis panels apply a restriction based on
the number of electricity contracts associated with
each building, retaining buildings with no more than
five contracts.

The final panel is used for:
- Household fixed-effects regressions.
- Event-study analyses of rooftop PV adoption.
- Load-shedding exposure analyses.
- PV capacity heterogeneity analyses.
- Robustness and misclassification sensitivity analyses.

---

## 3. Reproducibility Notes

The pipeline requires confidential electricity consumption, contract
location, building footprint, rooftop PV detection, registration, and
load-shedding datasets that are not distributed with this repository.

Steps 02b and 03 use separate historical intermediate datasets,
as documented in the path configuration above. Their numerical
equivalence to the outputs of Steps 02a and 02b has not been verified.

The complete standardized pipeline has not been rerun to establish
numerical equivalence with the reported analysis results.

---

## 4. Directory Contents

This directory contains:
- Numbered Python data-preparation scripts.
- This README documenting the processing sequence,
  dependencies, outputs, and provenance limitations.

Intermediate and final datasets are excluded from this public code repository.

---

## 5. Script Execution Order

The scripts are organized in the following processing order. Steps 02b and 03 require the historical intermediate inputs described above:

1. `01a_create_monthly_prepaid.py`
2. `01b_create_monthly_postpaid.py`
3. `02a_import_contract_locations.py`
4. `02b_merge_contracts_with_locations.py`
5. `03_assign_contracts_to_buildings.py`
6. `04_detect_shs_buildings_by_year.py`
7. `05a_match_contracts_to_shs_buildings.py`
8. `05b_apply_shs_classification_assumptions.py`
9. `05c_merge_shs_registration_data.py`
10. `06_assign_load_shedding_blocks.py`
11. `07a_calculate_monthly_load_shedding.py`
12. `08_build_final_analysis_panels.py`

Steps 03 and 04 are independent branches that both
feed into Step 05a.

Before rerunning the pipeline, verify that all external
inputs are available and that the configured paths
point to the intended datasets.

