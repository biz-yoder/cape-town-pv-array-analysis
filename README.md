# Rooftop Solar Adoption, Electricity Consumption, and Grid Reliability in Cape Town

This repository contains the data preparation and empirical analysis code for research examining how small-scale rooftop solar (SSEG/SHS) adoption affects residential electricity consumption, household resilience to load shedding, and municipal electricity revenue in Cape Town, South Africa.

The analysis combines household-level electricity consumption records, rooftop solar detections from high-resolution aerial imagery, solar registration records, building-level information, and electricity load-shedding data.

## Repository Structure

| Folder | Description |
|---|---|
| `01_DataPreparation/` | Constructs household-month electricity panels, matches electricity contracts to buildings and rooftop solar detections, incorporates registration data, and prepares load-shedding exposure variables. |
| `02_Regressions/` | Estimates household fixed-effects models of electricity consumption, rooftop solar capacity, and load-shedding exposure, including tariff-specific analyses. |
| `03_EventStudies/` | Estimates event-study effects of rooftop solar adoption using Callaway-style methods and evaluates pre-treatment equivalence. |
| `04_Robustness/` | Examines sensitivity to assumed rooftop solar installation timing and changes in sample composition. |
| `05_Revenue/` | Estimates municipal electricity revenue impacts of rooftop solar adoption by tariff and evaluates alternative adoption and registration scenarios. |

## Analysis Overview

### 1. Data Preparation

The data preparation pipeline constructs the household-month datasets used in the empirical analysis.

Key processing steps include:

- Importing and aggregating prepaid and postpaid electricity consumption records.
- Combining electricity contract information with household location records.
- Matching electricity contracts to building footprints.
- Detecting and aggregating rooftop solar installations from aerial imagery by year.
- Matching household electricity contracts to buildings with detected rooftop solar.
- Applying rooftop solar classification assumptions and incorporating municipal registration data.
- Assigning load-shedding blocks and calculating monthly load-shedding exposure.
- Formatting the final prepaid and postpaid analysis panels.

See `01_DataPreparation/README.md` for additional documentation of the processing pipeline, inputs, outputs, and quality-assurance checks.

### 2. Fixed-Effects Regressions

The regression analysis examines the relationship between rooftop solar adoption and household electricity consumption, accounting for household and time fixed effects.

Analyses include:

- Combined prepaid and postpaid household specifications.
- Postpaid rooftop solar capacity and outage exposure models.
- Prepaid tariff-specific capacity and outage models.
- Interactions between rooftop solar capacity and load-shedding exposure.

See `02_Regressions/`.

### 3. Event-Study Analysis

The event-study analysis examines changes in household electricity consumption following rooftop solar adoption.

The analysis includes:

- Callaway-style event-study estimation.
- Dose-response estimates by rooftop solar capacity.
- Pre-treatment equivalence assessments.

See `03_EventStudies/`.

### 4. Robustness and Timing Sensitivity

Because rooftop solar adoption dates are inferred from annual aerial imagery, the robustness analysis evaluates sensitivity to alternative assumed installation dates and changes in household cohort composition.

Analyses include:

- Alternative installation-date assignments.
- Capacity-specific event-study estimates.
- Common-calendar comparisons.
- Household cohort and composition diagnostics.

See `04_Robustness/`.

### 5. Municipal Revenue Analysis

The revenue analysis estimates the implications of rooftop solar adoption for municipal residential electricity revenue.

Analyses include:

- Revenue impacts by electricity tariff.
- Actual versus universal rooftop solar adoption scenarios.
- Revenue recovery under alternative registration and compliance assumptions.
- Tariff-specific revenue implications.

See `05_Revenue/`.

## Data and Reproducibility

The scripts were developed and executed in a Linux-based research computing environment using Python.

The analysis relies on municipal electricity consumption and tariff records, contract and registration data, building footprints, aerial imagery-derived rooftop solar detections, and load-shedding information.

Important reproducibility considerations:

- Input datasets are not included in this repository.
- Several scripts reference data stored in project-specific directories on the original research computing environment. File paths may need to be updated before execution elsewhere.
- Python package dependencies should be installed and configured for the individual scripts.
- Execution logs and selected intermediate analytical outputs are included where available.
- Household-level intermediate data are excluded from the public repository.

The scripts are organized by analytical stage rather than as a single automated execution pipeline. Consult the individual scripts and the data-preparation README for specific input and output requirements.

## Outputs

Selected analytical outputs are included alongside the scripts that generate them, including:

- Regression estimates and summary tables.
- Event-study estimates and figures.
- Pre-treatment equivalence results.
- Installation-timing sensitivity figures and tables.
- Revenue impact estimates by tariff and scenario.

## Research Context

This repository supports empirical research on the distributional implications of rooftop solar adoption, residential electricity demand, and electricity grid reliability in Cape Town, South Africa.

The analysis combines econometric methods, geospatial data processing, and electricity-sector analysis to examine how distributed energy resources interact with household electricity consumption and municipal electricity systems.
