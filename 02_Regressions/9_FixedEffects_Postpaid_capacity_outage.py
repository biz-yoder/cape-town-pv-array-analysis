import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
from scipy.stats import norm
from pathlib import Path


###########################################################################
# CONFIG
###########################################################################

POSTPAID_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/postpaid_cleaned_vector_noagg.parquet"
)

OUTPUT_DIRECTORY = Path(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "9_out_09182026"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

END_DATE = pd.Timestamp("2023-05-31")


###########################################################################
# LOAD POSTPAID DATA
###########################################################################

print("\n" + "=" * 90)
print("LOADING POSTPAID DATA")
print("=" * 90)

postpaid = pd.read_parquet(
    POSTPAID_FILE
)

print(
    "Rows loaded:",
    f"{len(postpaid):,}"
)

print(
    "Unique accounts loaded:",
    f"{postpaid['contract_account_hashed'].nunique():,}"
)

input_duplicate_rows = postpaid.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ],
    keep=False,
).sum()

print(
    "Input rows involved in duplicate account-months:",
    f"{input_duplicate_rows:,}"
)

combined = postpaid.copy()

print(
    "Number of exact duplicate rows:",
    f"{combined.duplicated().sum():,}"
)

combined = (
    combined
    .drop_duplicates()
    .copy()
)


###########################################################################
# BASIC SAMPLE RESTRICTIONS
###########################################################################

print("\n" + "=" * 90)
print("APPLYING SAMPLE RESTRICTIONS")
print("=" * 90)

combined["kwh"] = pd.to_numeric(
    combined["kwh"],
    errors="coerce",
)

combined = combined.loc[
    combined["kwh"].notna()
    & combined["kwh"].ge(0)
].copy()

combined["month_year"] = pd.to_datetime(
    combined["month_year"],
    errors="coerce",
)

combined = combined.loc[
    combined["month_year"].notna()
    & combined["month_year"].le(END_DATE)
].copy()

combined["Area_number"] = pd.to_numeric(
    combined["Area_number"],
    errors="coerce",
)

combined = combined.loc[
    combined["Area_number"].between(
        1,
        16,
    )
].copy()

# Confirm this is truly a postpaid-only sample.
if "Type" in combined.columns:

    combined = combined.loc[
        combined["Type"]
        .astype(str)
        .str.lower()
        .eq("postpaid")
    ].copy()

print(
    "Postpaid observations after restrictions:",
    f"{len(combined):,}"
)

print(
    "Unique postpaid households after restrictions:",
    f"{combined['contract_account_hashed'].nunique():,}"
)

###########################################################################
# USE STEP-8 PV CAPACITY
###########################################################################

print("\n" + "=" * 90)
print("USING STEP-8 PV CAPACITY")
print("=" * 90)

if "PV_kW" not in combined.columns:
    raise KeyError(
        "PV_kW is missing from the Step-8 postpaid file."
    )

combined["PV_kW"] = pd.to_numeric(
    combined["PV_kW"],
    errors="coerce",
)

combined["shs_label_numeric"] = pd.to_numeric(
    combined["shs_label_numeric"],
    errors="coerce",
)

# Negative capacity should never occur.
negative_capacity = (
    combined["PV_kW"].notna()
    & combined["PV_kW"].lt(0)
)

print(
    "Rows with negative PV capacity:",
    f"{negative_capacity.sum():,}"
)

if negative_capacity.any():
    raise ValueError(
        "Negative PV capacity values found."
    )

# Check PV observations BEFORE assigning zero capacity to non-PV observations.
pv_missing_capacity = (
    combined["shs_label_numeric"].eq(1)
    & combined["PV_kW"].isna()
)

print(
    "PV household-months with missing Step-8 PV_kW:",
    f"{pv_missing_capacity.sum():,}"
)

if pv_missing_capacity.any():
    raise ValueError(
        "Some PV household-months have missing Step-8 PV_kW."
    )

# Non-PV household-months have zero installed PV capacity.
non_pv = combined["shs_label_numeric"].eq(0)

combined.loc[
    non_pv & combined["PV_kW"].isna(),
    "PV_kW",
] = 0.0

# After the assignment, there should be no unexplained missing capacity.
remaining_missing_capacity = combined["PV_kW"].isna()

print(
    "Remaining rows with missing PV_kW:",
    f"{remaining_missing_capacity.sum():,}"
)

if remaining_missing_capacity.any():
    raise ValueError(
        "PV_kW remains missing after assigning zero capacity "
        "to non-PV household-months."
    )

print(
    "\nPV capacity summary before trimming:"
)

print(
    combined["PV_kW"].describe(
        percentiles=[
            0.90,
            0.95,
            0.99,
        ]
    )
)

print(
    "\nShare of household-months with positive PV capacity:"
)

print(
    (combined["PV_kW"] > 0).mean()
)

###########################################################################
# TRIM EXTREME PV CAPACITY VALUES
###########################################################################

print("\n" + "=" * 90)
print("WINSORIZING EXTREME PV CAPACITY VALUES")
print("=" * 90)

positive_capacity = combined.loc[
    combined["PV_kW"] > 0,
    "PV_kW",
]

if positive_capacity.empty:

    raise ValueError(
        "No positive PV capacity observations "
        "remain in the postpaid sample."
    )

print(
    "\nPositive-capacity quantiles:"
)

print(
    positive_capacity.quantile(
        [
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
        ]
    )
)

pv_upper_bound = (
    positive_capacity
    .quantile(
        0.99
    )
)

print(
    f"\n99th percentile of positive PV capacity: "
    f"{pv_upper_bound:.6f} kW"
)

# Match prepaid capacity specification:
# winsorize at the 99th percentile of positive capacity.
combined["PV_kW_trimmed"] = (
    combined["PV_kW"]
    .clip(
        lower=0,
        upper=pv_upper_bound,
    )
)

print(
    "\nTrimmed PV capacity summary:"
)

print(
    combined[
        "PV_kW_trimmed"
    ].describe(
        percentiles=[
            0.90,
            0.95,
            0.99,
        ]
    )
)


###########################################################################
# POSTPAID PV CAPACITY COUNTS
###########################################################################

print("\n" + "=" * 90)
print("POSTPAID PV CAPACITY COUNTS")
print("=" * 90)

combined["positive_pv"] = (
    combined[
        "PV_kW_trimmed"
    ] > 0
).astype(int)

print(
    "\nPostpaid PV-capacity observations:"
)

print(
    combined[
        "positive_pv"
    ].agg(
        positive_pv_observations="sum",
        positive_pv_share="mean",
        observations="count",
    )
)

print(
    "\nCapacity among positive-PV household-month observations:"
)

print(
    combined.loc[
        combined[
            "PV_kW_trimmed"
        ] > 0,
        "PV_kW_trimmed",
    ].agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
)

ever_pv = (
    combined
    .groupby(
        "contract_account_hashed"
    )[
        "positive_pv"
    ]
    .max()
)

print(
    "\nUnique postpaid households ever observed with positive PV:",
    f"{(ever_pv == 1).sum():,}"
)


###########################################################################
# CONSTRUCT OUTAGE VARIABLES
###########################################################################

print("\n" + "=" * 90)
print("CONSTRUCTING MODEL VARIABLES")
print("=" * 90)

combined[
    "total_duration_hr"
] = pd.to_numeric(
    combined[
        "total_duration_hr"
    ],
    errors="coerce",
)

# Convert outage hours to Stage-1-equivalent units.
combined[
    "total_duration_stage1"
] = (
    combined[
        "total_duration_hr"
    ]
    / 2.5
)

# Outage × continuous PV capacity.
combined[
    "outage_x_pv_capacity"
] = (
    combined[
        "total_duration_stage1"
    ]
    * combined[
        "PV_kW_trimmed"
    ]
)

CONTROL_VARS = [
    "total_duration_stage1",
    "PV_kW_trimmed",
    "outage_x_pv_capacity",
]

for variable in (
    CONTROL_VARS
    + [
        "kwh",
        "Area_number",
    ]
):

    combined[
        variable
    ] = pd.to_numeric(
        combined[
            variable
        ],
        errors="coerce",
    )

missing_before_drop = (
    combined[
        CONTROL_VARS
        + [
            "kwh",
            "Area_number",
            "contract_account_hashed",
            "month_year",
        ]
    ]
    .isna()
    .sum()
)

print(
    "\nMissing regression values before drop:"
)

print(
    missing_before_drop
)

combined = combined.dropna(
    subset=(
        CONTROL_VARS
        + [
            "kwh",
            "Area_number",
            "contract_account_hashed",
            "month_year",
        ]
    )
).copy()


###########################################################################
# DUPLICATE ACCOUNT-MONTH QA
###########################################################################

print("\n" + "=" * 90)
print("CHECKING DUPLICATE ACCOUNT-MONTHS")
print("=" * 90)

combined = (
    combined
    .sort_values(
        [
            "contract_account_hashed",
            "month_year",
        ]
    )
    .reset_index(
        drop=True
    )
)

panel_duplicates = combined.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ],
    keep=False,
)

print(
    "Rows involved in duplicate postpaid account-months:",
    f"{panel_duplicates.sum():,}"
)

duplicate_combinations = (
    combined.loc[
        panel_duplicates,
        [
            "contract_account_hashed",
            "month_year",
        ],
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    "Distinct duplicated postpaid account-months:",
    f"{duplicate_combinations:,}"
)

if panel_duplicates.any():

    duplicate_rows = (
        combined.loc[
            panel_duplicates
        ]
        .copy()
    )

    regression_relevant_cols = [
        "kwh",
        "Area_number",
        "PV_kW_trimmed",
        "total_duration_stage1",
        "outage_x_pv_capacity",
    ]

    duplicate_regression_variation = (
        duplicate_rows
        .groupby(
            [
                "contract_account_hashed",
                "month_year",
            ]
        )[
            regression_relevant_cols
        ]
        .nunique(
            dropna=False
        )
    )

    differences_by_variable = (
        duplicate_regression_variation
        .gt(1)
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nDuplicated postpaid account-months differing "
        "on regression variables:"
    )

    print(
        differences_by_variable[
            differences_by_variable > 0
        ]
    )

    if (
        differences_by_variable
        .gt(0)
        .any()
    ):

        raise ValueError(
            "Some duplicate postpaid account-months differ "
            "on variables used in the capacity regression."
        )

    duplicates_to_drop = (
        combined
        .duplicated(
            subset=[
                "contract_account_hashed",
                "month_year",
            ],
            keep="first",
        )
        .sum()
    )

    print(
        "\nDropping one copy of analytically identical "
        "duplicate account-month observations:",
        f"{duplicates_to_drop:,}"
    )

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "contract_account_hashed",
                "month_year",
            ],
            keep="first",
        )
        .copy()
    )

remaining_duplicates = (
    combined
    .duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ]
    )
    .sum()
)

print(
    "\nRemaining duplicate postpaid account-months:",
    f"{remaining_duplicates:,}"
)

if remaining_duplicates > 0:

    raise ValueError(
        "Duplicate postpaid account-month observations remain."
    )


###########################################################################
# FINAL SAMPLE QA
###########################################################################

print("\n" + "=" * 90)
print("FINAL SAMPLE QA")
print("=" * 90)

print(
    "Final observations:",
    f"{len(combined):,}"
)

print(
    "Final unique households:",
    f"{combined['contract_account_hashed'].nunique():,}"
)

print(
    "Final time periods:",
    f"{combined['month_year'].nunique():,}"
)

print(
    "Final load-shedding blocks:",
    sorted(
        combined[
            "Area_number"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
)

print(
    "\nStage-1 equivalent outage distribution:"
)

print(
    combined[
        "total_duration_stage1"
    ].describe()
)


###########################################################################
# UNIQUE-HOUSEHOLD MEDIAN PV CAPACITY
###########################################################################

print("\n" + "=" * 90)
print("UNIQUE-HOUSEHOLD POSTPAID PV CAPACITY")
print("=" * 90)

# First calculate each household's median positive PV capacity.
# This gives every PV household one representative system-size estimate
# without overweighting households observed for more months or relying
# on the household's maximum estimated capacity.

household_capacity = (
    combined.loc[
        combined["PV_kW_trimmed"] > 0,
        [
            "contract_account_hashed",
            "PV_kW_trimmed",
        ],
    ]
    .groupby(
        "contract_account_hashed",
        as_index=False,
    )["PV_kW_trimmed"]
    .median()
)

# Then summarize across unique PV households.
household_capacity_summary = (
    household_capacity["PV_kW_trimmed"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
)

median_pv_capacity = float(
    household_capacity_summary["median"]
)

print(
    "\nDistribution of household-level median positive PV capacity:"
)

print(
    household_capacity_summary
)

print(
    f"\nMedian PV capacity across unique postpaid PV households: "
    f"{median_pv_capacity:.6f} kW"
)

###########################################################################
# SET PANEL INDEX
###########################################################################

combined = (
    combined
    .set_index(
        [
            "contract_account_hashed",
            "month_year",
        ]
    )
    .sort_index()
)


###########################################################################
# ESTIMATE POSTPAID PV-CAPACITY MODEL
###########################################################################

print("\n" + "=" * 90)
print("ESTIMATING POSTPAID PV-CAPACITY FIXED-EFFECTS MODEL")
print("=" * 90)

dependent = np.log(
    combined[
        ["kwh"]
    ].astype(float)
    + 0.01
)

exog = (
    combined[
        CONTROL_VARS
    ]
    .astype(float)
)

block_clusters = (
    combined[
        ["Area_number"]
    ]
    .astype(int)
)

model = PanelOLS(
    dependent,
    exog,
    entity_effects=True,
    time_effects=True,
    check_rank=False,
    drop_absorbed=True,
)

results = model.fit(
    cov_type="clustered",
    clusters=block_clusters,
    low_memory=True,
)

print(
    results.summary
)


###########################################################################
# LINEAR-COMBINATION HELPERS
###########################################################################

def linear_combination(
    result,
    terms,
):

    weights = pd.Series(
        0.0,
        index=result.params.index,
    )

    for term, weight in (
        terms.items()
    ):

        if term not in weights.index:

            raise KeyError(
                f"Coefficient '{term}' was not estimated. "
                f"Available coefficients: "
                f"{list(weights.index)}"
            )

        weights.loc[
            term
        ] = weight

    estimate = float(
        weights
        @ result.params
    )

    variance = float(
        weights
        @ result.cov
        @ weights
    )

    # Protect against tiny negative floating-point values.
    variance = max(
        variance,
        0
    )

    standard_error = np.sqrt(
        variance
    )

    return (
        estimate,
        standard_error,
    )


def report_linear_effect(
    result,
    terms,
    effect_type,
    label,
    capacity_kw=np.nan,
):

    estimate, standard_error = (
        linear_combination(
            result,
            terms,
        )
    )

    if standard_error > 0:

        z_stat = (
            estimate
            / standard_error
        )

        p_value = (
            2
            * norm.sf(
                abs(z_stat)
            )
        )

    else:

        z_stat = np.nan
        p_value = np.nan

    lower = (
        estimate
        - 1.96
        * standard_error
    )

    upper = (
        estimate
        + 1.96
        * standard_error
    )

    percent_effect = (
        100
        * (
            np.exp(
                estimate
            )
            - 1
        )
    )

    lower_percent = (
        100
        * (
            np.exp(
                lower
            )
            - 1
        )
    )

    upper_percent = (
        100
        * (
            np.exp(
                upper
            )
            - 1
        )
    )

    print(
        f"{label}: "
        f"coef={estimate:.6f}, "
        f"SE={standard_error:.6f}, "
        f"p={p_value:.4f}, "
        f"effect={percent_effect:.3f}%, "
        f"95% CI=["
        f"{lower_percent:.3f}%, "
        f"{upper_percent:.3f}%]"
    )

    return {
        "sample": "Postpaid",
        "effect_type": effect_type,
        "effect": label,
        "capacity_kw": capacity_kw,
        "coefficient": estimate,
        "standard_error": standard_error,
        "p_value": p_value,
        "percent_effect": percent_effect,
        "lower_95_percent": lower_percent,
        "upper_95_percent": upper_percent,
    }


###########################################################################
# POSTPAID PV CAPACITY EFFECT
###########################################################################

print("\n" + "=" * 90)
print("POSTPAID PV CAPACITY EFFECT")
print("=" * 90)

capacity_effect = report_linear_effect(
    results,
    {
        "PV_kW_trimmed": 1,
    },
    "pv_capacity_effect_per_kw",
    "Effect of one additional kW of PV capacity",
    capacity_kw=1.0,
)


###########################################################################
# POSTPAID OUTAGE EFFECTS BY PV CAPACITY
###########################################################################

print("\n" + "=" * 90)
print("POSTPAID OUTAGE EFFECTS BY PV CAPACITY")
print("=" * 90)

# Continuous-capacity specification:
#
# Non-PV outage effect:
#   beta_outage
#
# Change in outage response per additional 1 kW:
#   beta_outage_x_capacity
#
# Outage effect at capacity c:
#   beta_outage
#   + c * beta_outage_x_capacity


###########################################################################
# NON-PV OUTAGE EFFECT
###########################################################################

non_pv_outage_effect = report_linear_effect(
    results,
    {
        "total_duration_stage1": 1,
    },
    "non_pv_outage_effect",
    "Non-PV outage effect (+2.5 hours)",
    capacity_kw=0.0,
)


###########################################################################
# CHANGE IN OUTAGE RESPONSE PER +1 kW
###########################################################################

outage_response_per_kw = report_linear_effect(
    results,
    {
        "outage_x_pv_capacity": 1,
    },
    "outage_response_change_per_kw",
    "Change in outage response per additional 1 kW PV",
    capacity_kw=1.0,
)


###########################################################################
# OUTAGE EFFECT AT 1 kW
###########################################################################

outage_effect_1kw = report_linear_effect(
    results,
    {
        "total_duration_stage1": 1,
        "outage_x_pv_capacity": 1,
    },
    "outage_effect_at_1kw_pv",
    "Outage effect at 1 kW PV",
    capacity_kw=1.0,
)


###########################################################################
# OUTAGE EFFECT AT MEDIAN POSTPAID PV CAPACITY
###########################################################################

outage_effect_median = report_linear_effect(
    results,
    {
        "total_duration_stage1": 1,
        "outage_x_pv_capacity": median_pv_capacity,
    },
    "outage_effect_at_median_pv",
    (
        "Outage effect at median PV capacity "
        f"({median_pv_capacity:.3f} kW)"
    ),
    capacity_kw=median_pv_capacity,
)


###########################################################################
# SAVE RESULTS
###########################################################################

print("\n" + "=" * 90)
print("SAVING OUTPUTS")
print("=" * 90)

capacity_effect_output = pd.DataFrame(
    [
        capacity_effect
    ]
)

outage_effects_output = pd.DataFrame(
    [
        non_pv_outage_effect,
        outage_response_per_kw,
        outage_effect_1kw,
        outage_effect_median,
    ]
)

household_capacity_output = pd.DataFrame(
    [
        {
            "sample": "Postpaid",
            "count": household_capacity_summary["count"],
            "mean": household_capacity_summary["mean"],
            "median": household_capacity_summary["median"],
            "std": household_capacity_summary["std"],
            "min": household_capacity_summary["min"],
            "max": household_capacity_summary["max"],
            "winsor_cutoff_kW": pv_upper_bound,
        }
    ]
)


###########################################################################
# OUTPUT PATHS
###########################################################################

capacity_effect_csv = (
    OUTPUT_DIRECTORY
    / "postpaid_capacity_effect_09182026.csv"
)

outage_effects_csv = (
    OUTPUT_DIRECTORY
    / "postpaid_capacity_outage_effects_09182026.csv"
)

household_capacity_csv = (
    OUTPUT_DIRECTORY
    / "postpaid_unique_household_capacity_summary_09182026.csv"
)

summary_txt = (
    OUTPUT_DIRECTORY
    / "postpaid_capacity_regression_summary_09182026.txt"
)


###########################################################################
# WRITE OUTPUT FILES
###########################################################################

capacity_effect_output.to_csv(
    capacity_effect_csv,
    index=False,
)

outage_effects_output.to_csv(
    outage_effects_csv,
    index=False,
)

household_capacity_output.to_csv(
    household_capacity_csv,
    index=False,
)

with open(
    summary_txt,
    "w",
) as output_file:

    output_file.write(
        str(
            results.summary
        )
    )


###########################################################################
# PRINT OUTPUT LOCATIONS
###########################################################################

print(
    "\nSaved postpaid PV-capacity effect to:"
)

print(
    capacity_effect_csv
)

print(
    "\nSaved postpaid capacity/outage effects to:"
)

print(
    outage_effects_csv
)

print(
    "\nSaved postpaid unique-household capacity summary to:"
)

print(
    household_capacity_csv
)

print(
    "\nSaved postpaid capacity regression summary to:"
)

print(
    summary_txt
)


###########################################################################
# FINAL COMPACT RESULTS
###########################################################################

print("\n" + "=" * 90)
print("FINAL COMPACT RESULTS")
print("=" * 90)

print(
    f"\nPositive-capacity 99th percentile "
    f"(winsorization cutoff): "
    f"{pv_upper_bound:.6f} kW"
)

print(
    f"\nMedian PV capacity across unique "
    f"postpaid PV households: "
    f"{median_pv_capacity:.6f} kW"
)

print(
    "\nPV capacity effect:"
)

print(
    capacity_effect_output[
        [
            "coefficient",
            "standard_error",
            "p_value",
            "percent_effect",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nOutage effects:"
)

print(
    outage_effects_output[
        [
            "effect_type",
            "capacity_kw",
            "coefficient",
            "standard_error",
            "p_value",
            "percent_effect",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nProcessing complete!"
)