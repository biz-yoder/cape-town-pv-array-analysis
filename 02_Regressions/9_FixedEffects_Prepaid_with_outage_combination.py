import pandas as pd
from linearmodels.panel import PanelOLS
import numpy as np
from scipy.stats import norm

prepaid = pd.read_parquet(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/prepaid_cleaned_vector_noagg.parquet"
)

postpaid = pd.read_parquet(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/postpaid_cleaned_vector_noagg.parquet"
)

# Stack prepaid and postpaid observations
combined = pd.concat([prepaid, postpaid], ignore_index=True)

print("Number of exact duplicate rows:")
print(combined.duplicated().sum())

combined = combined.drop_duplicates().copy()
combined = combined.loc[combined["kwh"] >= 0].copy()

# Restrict analysis period
combined["month_year"] = pd.to_datetime(combined["month_year"])
combined = combined.loc[
    combined["month_year"] <= pd.Timestamp("2023-05-31")
].copy()

# Keep municipal load-shedding blocks 1–16
combined = combined.loc[
    combined["Area_number"].between(1, 16)
].copy()

combined = combined[
    ~(combined["Type"] == "prepaid") |
    (combined["tariff_category"].notna())
].copy()

# Combine prepaid tariff categories and postpaid status
combined["tariff_or_postpaid"] = np.where(
    combined["Type"].eq("prepaid"),
    combined["tariff_category"],
    "postpaid"
)

combined = combined.loc[
    combined["Type"].eq("prepaid")
    & combined["tariff_category"].notna()
].copy()

combined["tariff_or_postpaid"] = combined["tariff_category"]

print("\nSHS observations by prepaid tariff:")
print(
    combined.groupby("tariff_or_postpaid")["shs_label_numeric"]
    .agg(
        treated_household_months="sum",
        treated_share="mean",
        observations="count"
    )
)

print("\nUnique households ever observed with SHS by tariff:")
print(
    combined.assign(
        ever_shs=combined.groupby(
            "contract_account_hashed"
        )["shs_label_numeric"].transform("max")
    )
    .loc[lambda x: x["ever_shs"].eq(1)]
    .groupby("tariff_or_postpaid")["contract_account_hashed"]
    .nunique()
)

# Convert outage hours to Stage 1 equivalents
combined["total_duration_stage1"] = (
    combined["total_duration_hr"] / 2.5
).fillna(0)

combined = combined.sort_values(
    ["contract_account_hashed", "month_year"]
).reset_index(drop=True)

###########################################################################
# CHECK TARIFF SWITCHING
###########################################################################

# Number of distinct tariff/status categories used by each account
tariff_switch_check = (
    combined.groupby("contract_account_hashed")["tariff_or_postpaid"]
    .nunique()
    .rename("n_tariffs")
)

print("\nNumber of accounts by number of tariff categories:")
print(tariff_switch_check.value_counts().sort_index())

switching_accounts = tariff_switch_check[
    tariff_switch_check > 1
].index

print(
    "\nNumber of accounts that ever change tariff/status:",
    len(switching_accounts)
)

print(
    "Share of accounts that ever change tariff/status:",
    len(switching_accounts)
    / combined["contract_account_hashed"].nunique()
)

###########################################################################
# DEFINE TARIFF CATEGORIES
###########################################################################

print("\nTariff categories:")
print(combined["tariff_or_postpaid"].value_counts(dropna=False))

# Set Lifeline as the reference group.
# Replace "Lifeline" below with the exact category spelling in your data.
reference_tariff = "LifeLine"

available_tariffs = sorted(
    combined["tariff_or_postpaid"].dropna().unique()
)

if reference_tariff not in available_tariffs:
    raise ValueError(
        f"Reference tariff '{reference_tariff}' is not in the data. "
        f"Available categories are: {available_tariffs}"
    )

tariff_categories = [
    reference_tariff
] + [
    tariff
    for tariff in available_tariffs
    if tariff != reference_tariff
]

combined["tariff_or_postpaid"] = pd.Categorical(
    combined["tariff_or_postpaid"],
    categories=tariff_categories
)

# The first category, Lifeline, is omitted
tariff_dummies = pd.get_dummies(
    combined["tariff_or_postpaid"],
    prefix="tariff",
    drop_first=True,
    dtype=float
)

combined = pd.concat([combined, tariff_dummies], axis=1)

###########################################################################
# CREATE MODEL VARIABLES
###########################################################################

# Outage × SHS
combined["outage_x_shs"] = (
    combined["total_duration_stage1"]
    * combined["shs_label_numeric"]
)

# Tariff-specific interactions.
#
# Lifeline is the omitted/reference tariff. Therefore:
#   total_duration_stage1
#       = non-PV outage effect for Lifeline
#   outage_x_shs
#       = additional PV outage response for Lifeline
#
# For Domestic and Home User, we add:
#   outage × tariff
#   SHS × tariff
#   outage × SHS × tariff
#
# This allows BOTH the non-PV outage response and the differential
# PV outage response to vary across prepaid tariffs.

shs_tariff_interactions = []
outage_tariff_interactions = []
outage_shs_tariff_interactions = []

for tariff_dummy in tariff_dummies.columns:
    shs_interaction_name = f"shs_x_{tariff_dummy}"
    outage_interaction_name = f"outage_x_{tariff_dummy}"
    triple_interaction_name = f"outage_x_shs_x_{tariff_dummy}"

    combined[shs_interaction_name] = (
        combined["shs_label_numeric"]
        * combined[tariff_dummy]
    )

    combined[outage_interaction_name] = (
        combined["total_duration_stage1"]
        * combined[tariff_dummy]
    )

    combined[triple_interaction_name] = (
        combined["total_duration_stage1"]
        * combined["shs_label_numeric"]
        * combined[tariff_dummy]
    )

    shs_tariff_interactions.append(shs_interaction_name)
    outage_tariff_interactions.append(outage_interaction_name)
    outage_shs_tariff_interactions.append(triple_interaction_name)

CONTROL_VARS = [
    "total_duration_stage1",
    "shs_label_numeric",
    "outage_x_shs",
] + (
    shs_tariff_interactions
    + outage_tariff_interactions
    + outage_shs_tariff_interactions
)

# Ensure all regression variables are numeric
for variable in CONTROL_VARS + ["kwh", "Area_number"]:
    combined[variable] = pd.to_numeric(
        combined[variable],
        errors="coerce"
    )

combined = combined.dropna(
    subset=CONTROL_VARS
    + ["kwh", "Area_number", "contract_account_hashed", "month_year"]
).copy()

panel_duplicates = combined.duplicated(
    subset=["contract_account_hashed", "month_year"],
    keep=False
)

print("\nDuplicate household-month rows:")
print(panel_duplicates.sum())

print("\nNumber of duplicated household-month combinations:")
print(
    combined.loc[
        panel_duplicates,
        ["contract_account_hashed", "month_year"]
    ].drop_duplicates().shape[0]
)

print(
    combined.loc[panel_duplicates]
    .groupby(["contract_account_hashed", "month_year"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)

duplicate_rows = combined.loc[panel_duplicates].copy()

duplicate_summary = (
    duplicate_rows
    .groupby(["contract_account_hashed", "month_year"])
    .agg(
        n_rows=("kwh", "size"),
        n_tariffs=("tariff_or_postpaid", "nunique"),
        total_kwh=("kwh", "sum"),
        min_kwh=("kwh", "min"),
        max_kwh=("kwh", "max")
    )
)

print("\nCharacteristics of duplicate contract-months:")
print(duplicate_summary["n_tariffs"].value_counts())

print(
    duplicate_rows[
        [
            "contract_account_hashed",
            "month_year",
            "tariff_or_postpaid",
            "kwh"
        ]
    ]
    .sort_values(
        ["contract_account_hashed", "month_year"]
    )
    .head(40)
    .to_string(index=False)
)

# Variables that must be identical for duplicates to be safely collapsed
regression_relevant_cols = [
    "kwh",
    "Area_number",
    "tariff_or_postpaid",
    "shs_label_numeric",
    "total_duration_stage1",
    "outage_x_shs",
] + (
    list(tariff_dummies.columns)
    + shs_tariff_interactions
    + outage_tariff_interactions
    + outage_shs_tariff_interactions
)

duplicate_regression_variation = (
    combined.loc[panel_duplicates]
    .groupby(["contract_account_hashed", "month_year"])[regression_relevant_cols]
    .nunique(dropna=False)
)

differences_by_variable = (
    duplicate_regression_variation.gt(1)
    .sum()
    .sort_values(ascending=False)
)

print("\nDuplicated contract-months differing in regression variables:")
print(differences_by_variable[differences_by_variable > 0])

duplicate_rows = combined.loc[panel_duplicates].copy()

all_column_variation = (
    duplicate_rows
    .groupby(["contract_account_hashed", "month_year"])
    .nunique(dropna=False)
)

columns_that_differ = (
    all_column_variation.gt(1)
    .sum()
    .sort_values(ascending=False)
)

print("\nColumns that differ within duplicate contract-months:")
print(columns_that_differ[columns_that_differ > 0])

if differences_by_variable.gt(0).any():
    raise ValueError(
        "Some duplicate contract-months differ on regression variables."
    )

print(
    "\nDropping one copy of duplicated contract-month observations:",
    combined.duplicated(
        subset=["contract_account_hashed", "month_year"]
    ).sum()
)

# Duplicate contract-month observations arise from multiple
# SHS geometry records. Because all regression variables are
# identical, retain one observation per contract-month.

combined = combined.drop_duplicates(
    subset=["contract_account_hashed", "month_year"],
    keep="first"
).copy()

remaining_duplicates = combined.duplicated(
    subset=["contract_account_hashed", "month_year"]
).sum()

print("\nRemaining duplicate contract-months:", remaining_duplicates)

assert remaining_duplicates == 0

if remaining_duplicates > 0:
    raise ValueError(
        "Duplicate contract-month observations remain."
    )

###########################################################################
# SET PANEL INDEX
###########################################################################

combined = combined.set_index(
    ["contract_account_hashed", "month_year"]
).sort_index()

###########################################################################
# ESTIMATE MODEL
###########################################################################

dependent = np.log(
    combined[["kwh"]].astype(float) + 0.01
)

exog = combined[CONTROL_VARS].astype(float)

# Clustering variable must align with the panel index
block_clusters = combined[["Area_number"]].astype(int)

model = PanelOLS(
    dependent,
    exog,
    entity_effects=True,
    time_effects=True,
    check_rank=False,
    drop_absorbed=True
)

results = model.fit(
    cov_type="clustered",
    clusters=block_clusters,
    low_memory=True
)

print(results.summary)

lifeline_effect = results.params["shs_label_numeric"]

domestic_effect = (
    results.params["shs_label_numeric"]
    + results.params["shs_x_tariff_Domestic"]
)

home_user_effect = (
    results.params["shs_label_numeric"]
    + results.params["shs_x_tariff_Home User"]
)

print("\nLifeLine SHS coefficient:", lifeline_effect)
print("Domestic SHS coefficient:", domestic_effect)
print("Home User SHS coefficient:", home_user_effect)

def linear_combination(result, terms):
    weights = pd.Series(0.0, index=result.params.index)

    for term, weight in terms.items():
        weights.loc[term] = weight

    estimate = float(weights @ result.params)
    variance = float(weights @ result.cov @ weights)
    standard_error = np.sqrt(variance)

    return estimate, standard_error


tariff_effects = {
    "LifeLine": {
        "shs_label_numeric": 1
    },
    "Domestic": {
        "shs_label_numeric": 1,
        "shs_x_tariff_Domestic": 1
    },
    "Home User": {
        "shs_label_numeric": 1,
        "shs_x_tariff_Home User": 1
    }
}

print("\nTariff-specific SHS effects:")

for tariff, terms in tariff_effects.items():
    estimate, standard_error = linear_combination(results, terms)

    lower = estimate - 1.96 * standard_error
    upper = estimate + 1.96 * standard_error

    # Exact percentage transformation for a log outcome
    percent_effect = 100 * (np.exp(estimate) - 1)
    lower_percent = 100 * (np.exp(lower) - 1)
    upper_percent = 100 * (np.exp(upper) - 1)

    print(
        f"{tariff}: "
        f"coef={estimate:.4f}, "
        f"SE={standard_error:.4f}, "
        f"effect={percent_effect:.2f}%, "
        f"95% CI=[{lower_percent:.2f}%, {upper_percent:.2f}%]"
    )

###########################################################################
# TARIFF-SPECIFIC OUTAGE EFFECTS FOR NON-PV AND PV HOUSEHOLDS
###########################################################################

def report_linear_effect(result, terms, label, tariff, effect_type):
    """
    Report a linear combination of regression coefficients with:
      - coefficient
      - clustered standard error using the full covariance matrix
      - two-sided p-value
      - exact percentage effect for the log outcome
      - 95% confidence interval transformed to percent
    """
    estimate, standard_error = linear_combination(
        result,
        terms,
    )

    z_stat = (
        estimate
        / standard_error
    )

    p_value = (
        2
        * (
            1
            - norm.cdf(
                abs(z_stat)
            )
        )
    )

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
        f"95% CI=[{lower_percent:.3f}%, {upper_percent:.3f}%]"
    )

    return {
        "tariff": tariff,
        "effect_type": effect_type,
        "effect": label,
        "coef": estimate,
        "se": standard_error,
        "p_value": p_value,
        "percent_effect": percent_effect,
        "ci_lower_percent": lower_percent,
        "ci_upper_percent": upper_percent,
    }


# Terms for the three tariff-specific quantities we care about:
#
# 1. Non-PV outage effect
#    = outage response when SHS = 0
#
# 2. PV x outage differential
#    = how much the outage response differs for PV households
#
# 3. Total PV outage effect
#    = non-PV outage effect + PV x outage differential
#
# Lifeline is the reference category.

tariff_outage_terms = {
    "LifeLine": {
        "non_pv": {
            "total_duration_stage1": 1,
        },
        "pv_differential": {
            "outage_x_shs": 1,
        },
        "pv_total": {
            "total_duration_stage1": 1,
            "outage_x_shs": 1,
        },
    },

    "Domestic": {
        "non_pv": {
            "total_duration_stage1": 1,
            "outage_x_tariff_Domestic": 1,
        },
        "pv_differential": {
            "outage_x_shs": 1,
            "outage_x_shs_x_tariff_Domestic": 1,
        },
        "pv_total": {
            "total_duration_stage1": 1,
            "outage_x_tariff_Domestic": 1,
            "outage_x_shs": 1,
            "outage_x_shs_x_tariff_Domestic": 1,
        },
    },

    "Home User": {
        "non_pv": {
            "total_duration_stage1": 1,
            "outage_x_tariff_Home User": 1,
        },
        "pv_differential": {
            "outage_x_shs": 1,
            "outage_x_shs_x_tariff_Home User": 1,
        },
        "pv_total": {
            "total_duration_stage1": 1,
            "outage_x_tariff_Home User": 1,
            "outage_x_shs": 1,
            "outage_x_shs_x_tariff_Home User": 1,
        },
    },
}


print("\n" + "=" * 90)
print("TARIFF-SPECIFIC OUTAGE EFFECTS BY PV STATUS")
print("=" * 90)

outage_effect_rows = []

for tariff, term_sets in tariff_outage_terms.items():
    print("\n" + "-" * 90)
    print(tariff)
    print("-" * 90)

    non_pv_result = report_linear_effect(
        results,
        term_sets["non_pv"],
        "Non-PV outage effect",
        tariff,
        "non_pv_outage_effect",
    )

    pv_differential_result = report_linear_effect(
        results,
        term_sets["pv_differential"],
        "PV x outage differential",
        tariff,
        "pv_outage_differential",
    )

    pv_total_result = report_linear_effect(
        results,
        term_sets["pv_total"],
        "Total PV outage effect",
        tariff,
        "pv_total_outage_effect",
    )

    outage_effect_rows.extend(
        [
            non_pv_result,
            pv_differential_result,
            pv_total_result,
        ]
    )


###########################################################################
# SAVE TARIFF-SPECIFIC OUTAGE EFFECTS
###########################################################################

outage_effects_output = pd.DataFrame(
    outage_effect_rows
)

outage_effects_file = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "9_out_09182026/"
    "prepaid_tariff_specific_outage_effects_09182026.csv"
)

outage_effects_output.to_csv(
    outage_effects_file,
    index=False,
)

print(
    "\nSaved prepaid tariff-specific outage effects to:",
    outage_effects_file,
)

