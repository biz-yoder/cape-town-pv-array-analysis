import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import POSTPAID_PANEL, PREPAID_PANEL, REGRESSION_DIR

import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
from scipy.stats import norm


###############################################################################
# CONFIG
###############################################################################

PREPAID_FILE = (
    str(PREPAID_PANEL)
)

OUTPUT_DIRECTORY = Path(
    REGRESSION_DIR
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

END_DATE = pd.Timestamp("2023-05-31")


###############################################################################
# LOAD CURRENT PREPAID DATA
###############################################################################

print("\n" + "=" * 90)
print("LOADING CURRENT PREPAID DATA")
print("=" * 90)

prepaid = pd.read_parquet(
    PREPAID_FILE
)

print(
    "Rows loaded:",
    f"{len(prepaid):,}"
)

print(
    "Unique accounts loaded:",
    f"{prepaid['contract_account_hashed'].nunique():,}"
)

input_duplicate_rows = prepaid.duplicated(
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

combined = prepaid.copy()

print(
    "Number of exact duplicate rows:",
    f"{combined.duplicated().sum():,}"
)

combined = combined.drop_duplicates().copy()


###############################################################################
# BASIC SAMPLE RESTRICTIONS
###############################################################################

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

# Municipal load-shedding blocks used in the regressions.
combined = combined.loc[
    combined["Area_number"].between(1, 16)
].copy()

# Confirm prepaid-only sample.
if "Type" in combined.columns:
    combined = combined.loc[
        combined["Type"]
        .astype(str)
        .str.lower()
        .eq("prepaid")
    ].copy()

# Require known tariff.
combined = combined.loc[
    combined["tariff_category"].notna()
].copy()

combined["tariff_or_postpaid"] = (
    combined["tariff_category"]
)

print(
    "Prepaid observations after restrictions:",
    f"{len(combined):,}"
)

print(
    "Unique prepaid households after restrictions:",
    f"{combined['contract_account_hashed'].nunique():,}"
)

print(
    "\nTariff categories after restrictions:"
)

print(
    combined[
        "tariff_or_postpaid"
    ].value_counts(
        dropna=False
    )
)


###############################################################################
# USE STEP-8 PV CAPACITY
###############################################################################

print("\n" + "=" * 90)
print("USING STEP-8 PV CAPACITY")
print("=" * 90)

# Step 8 already constructs PV_kW from PV_capacity_W_per_contract and then
# backfills PV_kW for the inferred pre-detection PV period. Use that saved
# Step-8 analysis variable directly so the continuous-capacity regressions are
# consistent with the binary PV backfill.
if "PV_kW" not in combined.columns:
    raise KeyError(
        "PV_kW is missing from the Step-8 prepaid file."
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

# There should now be no unexplained missing capacity.
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

###############################################################################
# WINSORIZE EXTREME PV CAPACITY
###############################################################################

print("\n" + "=" * 90)
print("WINSORIZING PV CAPACITY")
print("=" * 90)

positive_capacity = combined.loc[
    combined["PV_kW"] > 0,
    "PV_kW",
]

if positive_capacity.empty:
    raise ValueError(
        "No positive PV capacity observations "
        "remain in the prepaid sample."
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
    positive_capacity.quantile(
        0.99
    )
)

print(
    "\n99th percentile of positive "
    f"PV capacity: {pv_upper_bound:.6f} kW"
)

# Winsorize rather than drop.
combined["PV_kW_trimmed"] = (
    combined["PV_kW"].clip(
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


###############################################################################
# PV COUNTS AND CAPACITY BY TARIFF
###############################################################################

print("\n" + "=" * 90)
print("PV CAPACITY COUNTS BY TARIFF")
print("=" * 90)

combined["positive_pv"] = (
    combined["PV_kW_trimmed"] > 0
).astype(int)

print(
    "\nPV observations by prepaid tariff:"
)

print(
    combined.groupby(
        "tariff_or_postpaid"
    ).agg(
        observations=(
            "contract_account_hashed",
            "size",
        ),
        positive_pv_observations=(
            "positive_pv",
            "sum",
        ),
        positive_pv_share=(
            "positive_pv",
            "mean",
        ),
        mean_capacity_all=(
            "PV_kW_trimmed",
            "mean",
        ),
    )
)

print(
    "\nCapacity among positive-PV "
    "household-month observations:"
)

print(
    combined.loc[
        combined["PV_kW_trimmed"] > 0
    ]
    .groupby(
        "tariff_or_postpaid"
    )["PV_kW_trimmed"]
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


###############################################################################
# REPRESENTATIVE PV CAPACITY
###############################################################################

# Representative household-level capacities are calculated later, after all
# regression-sample restrictions have been applied. For each household-tariff
# combination, the script uses the median positive trimmed PV capacity, and
# then summarizes across unique household-tariff combinations within tariff.


###############################################################################
# OUTAGE VARIABLES
###############################################################################

print("\n" + "=" * 90)
print("CONSTRUCTING OUTAGE VARIABLES")
print("=" * 90)

combined["total_duration_hr"] = pd.to_numeric(
    combined["total_duration_hr"],
    errors="coerce",
)

combined["total_duration_stage1"] = (
    combined["total_duration_hr"]
    / 2.5
)

# Step 7 should already contain zero for valid block-months with no scheduled
# outage. Do not automatically turn genuinely missing outage data into zero.
combined["outage_x_pv_capacity"] = (
    combined["total_duration_stage1"]
    * combined["PV_kW_trimmed"]
)


###############################################################################
# TARIFF SWITCHING QA
###############################################################################

print("\n" + "=" * 90)
print("CHECKING TARIFF SWITCHING")
print("=" * 90)

tariff_switch_check = (
    combined.groupby(
        "contract_account_hashed"
    )["tariff_or_postpaid"]
    .nunique()
    .rename("n_tariffs")
)

print(
    "\nNumber of accounts by "
    "number of tariff categories:"
)

print(
    tariff_switch_check
    .value_counts()
    .sort_index()
)

switching_accounts = (
    tariff_switch_check.loc[
        tariff_switch_check > 1
    ].index
)

print(
    "\nNumber of accounts "
    "that ever change tariff:",
    f"{len(switching_accounts):,}",
)

print(
    "Share of accounts "
    "that ever change tariff:",
    len(switching_accounts)
    / combined[
        "contract_account_hashed"
    ].nunique(),
)


###############################################################################
# DEFINE TARIFF CATEGORIES
###############################################################################

print("\n" + "=" * 90)
print("DEFINING TARIFF CATEGORIES")
print("=" * 90)

reference_tariff = "LifeLine"

available_tariffs = sorted(
    combined[
        "tariff_or_postpaid"
    ]
    .dropna()
    .unique()
)

print(
    "Available tariffs:",
    available_tariffs,
)

if reference_tariff not in available_tariffs:

    raise ValueError(
        f"Reference tariff "
        f"'{reference_tariff}' "
        f"is not in the data. "
        f"Available categories are: "
        f"{available_tariffs}"
    )

# Restrict explicitly to the three tariffs analyzed in the paper.
analysis_tariffs = [
    "LifeLine",
    "Domestic",
    "Home User",
]

missing_tariffs = [
    tariff
    for tariff in analysis_tariffs
    if tariff not in available_tariffs
]

if missing_tariffs:
    raise ValueError(
        "Expected prepaid tariff categories "
        f"not found: {missing_tariffs}. "
        f"Available tariffs: {available_tariffs}"
    )

combined = combined.loc[
    combined[
        "tariff_or_postpaid"
    ].isin(
        analysis_tariffs
    )
].copy()

tariff_categories = [
    "LifeLine",
    "Domestic",
    "Home User",
]

combined[
    "tariff_or_postpaid"
] = pd.Categorical(
    combined[
        "tariff_or_postpaid"
    ],
    categories=tariff_categories,
)

tariff_dummies = pd.get_dummies(
    combined[
        "tariff_or_postpaid"
    ],
    prefix="tariff",
    drop_first=True,
    dtype=float,
)

combined = pd.concat(
    [
        combined,
        tariff_dummies,
    ],
    axis=1,
)

print(
    "\nTariff dummy columns:"
)

print(
    list(
        tariff_dummies.columns
    )
)


###############################################################################
# CREATE CAPACITY MODEL VARIABLES
###############################################################################

print("\n" + "=" * 90)
print("CREATING REGRESSION VARIABLES")
print("=" * 90)

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

capacity_tariff_interactions = []
outage_tariff_interactions = []
outage_capacity_tariff_interactions = []

for tariff_dummy in tariff_dummies.columns:

    capacity_interaction_name = (
        f"pvkw_x_{tariff_dummy}"
    )

    outage_interaction_name = (
        f"outage_x_{tariff_dummy}"
    )

    triple_interaction_name = (
        f"outage_x_pvkw_x_{tariff_dummy}"
    )

    combined[
        capacity_interaction_name
    ] = (
        combined["PV_kW_trimmed"]
        * combined[tariff_dummy]
    )

    combined[
        outage_interaction_name
    ] = (
        combined[
            "total_duration_stage1"
        ]
        * combined[tariff_dummy]
    )

    combined[
        triple_interaction_name
    ] = (
        combined[
            "total_duration_stage1"
        ]
        * combined[
            "PV_kW_trimmed"
        ]
        * combined[tariff_dummy]
    )

    capacity_tariff_interactions.append(
        capacity_interaction_name
    )

    outage_tariff_interactions.append(
        outage_interaction_name
    )

    outage_capacity_tariff_interactions.append(
        triple_interaction_name
    )


CONTROL_VARS = [
    "total_duration_stage1",
    "PV_kW_trimmed",
    "outage_x_pv_capacity",
] + (
    capacity_tariff_interactions
    + outage_tariff_interactions
    + outage_capacity_tariff_interactions
)

print(
    "\nRegression variables:"
)

for variable in CONTROL_VARS:
    print(variable)


###############################################################################
# ENSURE NUMERIC / DROP MISSING MODEL VALUES
###############################################################################

print("\n" + "=" * 90)
print("CHECKING MISSING REGRESSION VALUES")
print("=" * 90)

for variable in (
    CONTROL_VARS
    + [
        "kwh",
        "Area_number",
    ]
):

    combined[variable] = pd.to_numeric(
        combined[variable],
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


###############################################################################
# DUPLICATE CONTRACT-MONTH QA
###############################################################################

print("\n" + "=" * 90)
print("CHECKING DUPLICATE CONTRACT-MONTHS")
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
    "Rows belonging to duplicate "
    "contract-months:",
    f"{panel_duplicates.sum():,}",
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
    "Number of duplicated "
    "contract-month combinations:",
    f"{duplicate_combinations:,}",
)

if panel_duplicates.any():

    duplicate_rows = (
        combined.loc[
            panel_duplicates
        ].copy()
    )

    regression_relevant_cols = [
        "kwh",
        "Area_number",
        "tariff_or_postpaid",
        "PV_kW_trimmed",
        "total_duration_stage1",
        "outage_x_pv_capacity",
    ] + (
        list(
            tariff_dummies.columns
        )
        + capacity_tariff_interactions
        + outage_tariff_interactions
        + outage_capacity_tariff_interactions
    )

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
        "\nDuplicated contract-months "
        "differing in regression variables:"
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
            "Some duplicate contract-months "
            "differ on variables used in "
            "the capacity regression."
        )

    n_to_drop = combined.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep="first",
    ).sum()

    print(
        "\nDropping analytically identical "
        "duplicate rows:",
        f"{n_to_drop:,}",
    )

    combined = combined.drop_duplicates(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep="first",
    ).copy()


remaining_duplicates = combined.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ]
).sum()

print(
    "\nRemaining duplicate contract-months:",
    f"{remaining_duplicates:,}",
)

if remaining_duplicates > 0:

    raise ValueError(
        "Duplicate contract-month "
        "observations remain."
    )


###############################################################################
# FINAL SAMPLE QA
###############################################################################

print("\n" + "=" * 90)
print("FINAL SAMPLE QA")
print("=" * 90)

print(
    "Final observations:",
    f"{len(combined):,}",
)

print(
    "Final unique households:",
    f"{combined['contract_account_hashed'].nunique():,}",
)

print(
    "Final time periods:",
    f"{combined['month_year'].nunique():,}",
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
    ),
)

print(
    "\nFinal tariff observations:"
)

print(
    combined[
        "tariff_or_postpaid"
    ].value_counts()
)

print(
    "\nStage-1 equivalent "
    "outage distribution:"
)

print(
    combined[
        "total_duration_stage1"
    ].describe()
)


###############################################################################
# RECOMPUTE UNIQUE-HOUSEHOLD MEDIANS IN FINAL REGRESSION SAMPLE
###############################################################################

print("\n" + "=" * 90)
print("FINAL-SAMPLE UNIQUE-HOUSEHOLD PV CAPACITY")
print("=" * 90)

# Recalculate after ALL regression sample restrictions so the medians
# correspond exactly to the estimated model sample.

final_household_tariff_capacity = (
    combined.loc[
        combined["PV_kW_trimmed"] > 0,
        [
            "contract_account_hashed",
            "tariff_or_postpaid",
            "PV_kW_trimmed",
        ],
    ]
    .groupby(
        [
            "contract_account_hashed",
            "tariff_or_postpaid",
        ],
        observed=True,
        as_index=False,
    )["PV_kW_trimmed"]
    .median()
)

final_household_capacity_summary = (
    final_household_tariff_capacity
    .groupby(
        "tariff_or_postpaid",
        observed=True,
    )["PV_kW_trimmed"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
)

print(
    final_household_capacity_summary
)

median_capacity_by_tariff = (
    final_household_capacity_summary[
        "median"
    ].to_dict()
)

print(
    "\nFinal tariff-specific median "
    "PV capacities:"
)

for tariff in [
    "LifeLine",
    "Domestic",
    "Home User",
]:

    print(
        f"{tariff}: "
        f"{median_capacity_by_tariff[tariff]:.6f} kW"
    )


###############################################################################
# SET PANEL INDEX
###############################################################################

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


###############################################################################
# ESTIMATE CAPACITY MODEL
###############################################################################

print("\n" + "=" * 90)
print("ESTIMATING PREPAID PV-CAPACITY FIXED-EFFECTS MODEL")
print("=" * 90)

dependent = np.log(
    combined[
        ["kwh"]
    ].astype(float)
    + 0.01
)

exog = combined[
    CONTROL_VARS
].astype(float)

block_clusters = combined[
    ["Area_number"]
].astype(int)

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


###############################################################################
# LINEAR COMBINATION HELPERS
###############################################################################

def linear_combination(
    result,
    terms,
):

    weights = pd.Series(
        0.0,
        index=result.params.index,
    )

    for term, weight in terms.items():

        if term not in weights.index:

            raise KeyError(
                f"Coefficient '{term}' "
                f"was not estimated. "
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

    variance = max(
        variance,
        0,
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
    tariff,
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
        "tariff": tariff,
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


###############################################################################
# TARIFF-SPECIFIC EFFECT OF +1 kW PV CAPACITY
###############################################################################

print("\n" + "=" * 90)
print("TARIFF-SPECIFIC EFFECT OF ONE ADDITIONAL kW PV")
print("=" * 90)

tariff_capacity_effects = {

    "LifeLine": {
        "PV_kW_trimmed": 1,
    },

    "Domestic": {
        "PV_kW_trimmed": 1,
        "pvkw_x_tariff_Domestic": 1,
    },

    "Home User": {
        "PV_kW_trimmed": 1,
        "pvkw_x_tariff_Home User": 1,
    },
}

capacity_effect_rows = []

for tariff, terms in (
    tariff_capacity_effects.items()
):

    result_row = report_linear_effect(
        results,
        terms,
        tariff,
        "pv_capacity_effect_per_kw",
        (
            "Effect of one additional "
            f"kW PV capacity: {tariff}"
        ),
        capacity_kw=1.0,
    )

    capacity_effect_rows.append(
        result_row
    )


###############################################################################
# TARIFF-SPECIFIC OUTAGE COMPONENTS
###############################################################################

tariff_outage_terms = {

    "LifeLine": {
        "non_pv": {
            "total_duration_stage1": 1,
        },
        "per_kw": {
            "outage_x_pv_capacity": 1,
        },
    },

    "Domestic": {
        "non_pv": {
            "total_duration_stage1": 1,
            "outage_x_tariff_Domestic": 1,
        },
        "per_kw": {
            "outage_x_pv_capacity": 1,
            "outage_x_pvkw_x_tariff_Domestic": 1,
        },
    },

    "Home User": {
        "non_pv": {
            "total_duration_stage1": 1,
            "outage_x_tariff_Home User": 1,
        },
        "per_kw": {
            "outage_x_pv_capacity": 1,
            "outage_x_pvkw_x_tariff_Home User": 1,
        },
    },
}


###############################################################################
# REPORT OUTAGE EFFECTS
###############################################################################

print("\n" + "=" * 90)
print("TARIFF-SPECIFIC OUTAGE EFFECTS BY PV CAPACITY")
print("=" * 90)

outage_capacity_rows = []

for tariff, term_sets in (
    tariff_outage_terms.items()
):

    print(
        "\n" + "-" * 90
    )

    print(
        tariff
    )

    print(
        "-" * 90
    )

    # ---------------------------------------------------------
    # Non-PV outage effect
    # ---------------------------------------------------------

    non_pv_result = (
        report_linear_effect(
            results,
            term_sets[
                "non_pv"
            ],
            tariff,
            "non_pv_outage_effect",
            (
                "Non-PV outage effect "
                "(+2.5 hours)"
            ),
            capacity_kw=0.0,
        )
    )

    outage_capacity_rows.append(
        non_pv_result
    )


    # ---------------------------------------------------------
    # Change in outage response per +1 kW
    # ---------------------------------------------------------

    per_kw_result = (
        report_linear_effect(
            results,
            term_sets[
                "per_kw"
            ],
            tariff,
            "outage_response_change_per_kw",
            (
                "Change in outage response "
                "per additional 1 kW PV"
            ),
            capacity_kw=1.0,
        )
    )

    outage_capacity_rows.append(
        per_kw_result
    )


    # ---------------------------------------------------------
    # Outage effect at 1 kW
    # ---------------------------------------------------------

    one_kw_terms = {}

    for term, weight in (
        term_sets[
            "non_pv"
        ].items()
    ):

        one_kw_terms[
            term
        ] = (
            one_kw_terms.get(
                term,
                0,
            )
            + weight
        )

    for term, weight in (
        term_sets[
            "per_kw"
        ].items()
    ):

        one_kw_terms[
            term
        ] = (
            one_kw_terms.get(
                term,
                0,
            )
            + weight
        )

    one_kw_result = (
        report_linear_effect(
            results,
            one_kw_terms,
            tariff,
            "outage_effect_at_1kw_pv",
            (
                "Outage effect at "
                "1 kW PV"
            ),
            capacity_kw=1.0,
        )
    )

    outage_capacity_rows.append(
        one_kw_result
    )


    # ---------------------------------------------------------
    # Outage effect at tariff-specific median capacity
    # ---------------------------------------------------------

    median_kw = float(
        median_capacity_by_tariff[
            tariff
        ]
    )

    median_terms = {}

    for term, weight in (
        term_sets[
            "non_pv"
        ].items()
    ):

        median_terms[
            term
        ] = (
            median_terms.get(
                term,
                0,
            )
            + weight
        )

    for term, weight in (
        term_sets[
            "per_kw"
        ].items()
    ):

        median_terms[
            term
        ] = (
            median_terms.get(
                term,
                0,
            )
            + median_kw
            * weight
        )

    median_result = (
        report_linear_effect(
            results,
            median_terms,
            tariff,
            "outage_effect_at_median_pv",
            (
                "Outage effect at "
                f"median PV capacity "
                f"({median_kw:.3f} kW)"
            ),
            capacity_kw=median_kw,
        )
    )

    outage_capacity_rows.append(
        median_result
    )


###############################################################################
# SAVE OUTPUTS
###############################################################################

print("\n" + "=" * 90)
print("SAVING OUTPUTS")
print("=" * 90)

capacity_effects_df = pd.DataFrame(
    capacity_effect_rows
)

outage_capacity_effects_df = pd.DataFrame(
    outage_capacity_rows
)

median_capacity_output = (
    final_household_capacity_summary
    .reset_index()
)

median_capacity_output[
    "winsor_cutoff_kW"
] = pv_upper_bound


capacity_effect_csv = (
    OUTPUT_DIRECTORY
    / "prepaid_tariff_capacity_effects_currentworkflow_09182026.csv"
)

outage_effect_csv = (
    OUTPUT_DIRECTORY
    / "prepaid_tariff_capacity_outage_effects_currentworkflow_09182026.csv"
)

median_capacity_csv = (
    OUTPUT_DIRECTORY
    / "prepaid_unique_household_capacity_summary_currentworkflow_09182026.csv"
)

summary_txt = (
    OUTPUT_DIRECTORY
    / "prepaid_tariff_capacity_regression_summary_currentworkflow_09182026.txt"
)


capacity_effects_df.to_csv(
    capacity_effect_csv,
    index=False,
)

outage_capacity_effects_df.to_csv(
    outage_effect_csv,
    index=False,
)

median_capacity_output.to_csv(
    median_capacity_csv,
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


print(
    "\nSaved tariff-specific "
    "capacity effects to:"
)

print(
    capacity_effect_csv
)

print(
    "\nSaved tariff-specific "
    "capacity/outage effects to:"
)

print(
    outage_effect_csv
)

print(
    "\nSaved unique-household "
    "capacity summary to:"
)

print(
    median_capacity_csv
)

print(
    "\nSaved regression summary to:"
)

print(
    summary_txt
)


###############################################################################
# FINAL COMPACT RESULTS
###############################################################################

print("\n" + "=" * 90)
print("FINAL COMPACT RESULTS")
print("=" * 90)

print(
    f"\nWinsor cutoff: "
    f"{pv_upper_bound:.6f} kW"
)

print(
    "\nMedian PV capacity across "
    "unique PV household-tariff combinations:"
)

for tariff in [
    "LifeLine",
    "Domestic",
    "Home User",
]:

    print(
        f"  {tariff}: "
        f"{median_capacity_by_tariff[tariff]:.3f} kW"
    )

print(
    "\nPV capacity effects:"
)

print(
    capacity_effects_df[
        [
            "tariff",
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
    outage_capacity_effects_df[
        [
            "tariff",
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