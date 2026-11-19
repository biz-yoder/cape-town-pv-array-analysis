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

print("Rows loaded:", f"{len(postpaid):,}")
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
    & combined["month_year"].le(
        END_DATE
    )
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
# CONSTRUCT MODEL VARIABLES
###########################################################################

print("\n" + "=" * 90)
print("CONSTRUCTING MODEL VARIABLES")
print("=" * 90)

combined["shs_label_numeric"] = pd.to_numeric(
    combined["shs_label_numeric"],
    errors="coerce",
)

combined["total_duration_hr"] = pd.to_numeric(
    combined["total_duration_hr"],
    errors="coerce",
)

# Convert outage hours to Stage-1-equivalent units.
combined["total_duration_stage1"] = (
    combined["total_duration_hr"] / 2.5
)

# Outage × SHS interaction.
combined["outage_x_shs"] = (
    combined["total_duration_stage1"]
    * combined["shs_label_numeric"]
)

CONTROL_VARS = [
    "total_duration_stage1",
    "shs_label_numeric",
    "outage_x_shs",
]

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
    "\nMissing regression values before drop:"
)

print(
    missing_before_drop
)

combined = combined.dropna(
    subset=
    CONTROL_VARS
    + [
        "kwh",
        "Area_number",
        "contract_account_hashed",
        "month_year",
    ]
).copy()


###########################################################################
# DESCRIPTIVE SHS COUNTS
###########################################################################

print("\n" + "=" * 90)
print("POSTPAID SHS COUNTS")
print("=" * 90)

print(
    combined[
        "shs_label_numeric"
    ].agg(
        treated_household_months="sum",
        treated_share="mean",
        observations="count",
    )
)

ever_shs = (
    combined
    .groupby(
        "contract_account_hashed"
    )[
        "shs_label_numeric"
    ]
    .max()
)

print(
    "\nUnique postpaid households ever observed with SHS:",
    f"{(ever_shs == 1).sum():,}"
)


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
    duplicate_rows = combined.loc[
        panel_duplicates
    ].copy()

    regression_relevant_cols = [
        "kwh",
        "Area_number",
        "shs_label_numeric",
        "total_duration_stage1",
        "outage_x_shs",
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

    all_column_variation = (
        duplicate_rows
        .groupby(
            [
                "contract_account_hashed",
                "month_year",
            ]
        )
        .nunique(
            dropna=False
        )
    )

    columns_that_differ = (
        all_column_variation
        .gt(1)
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nAll columns that differ within duplicate account-months:"
    )

    print(
        columns_that_differ[
            columns_that_differ > 0
        ]
    )

    if differences_by_variable.gt(0).any():
        print(
            "\nExamples of problematic duplicate account-months:"
        )

        problematic_keys = (
            duplicate_regression_variation
            .loc[
                duplicate_regression_variation
                .gt(1)
                .any(axis=1)
            ]
            .head(10)
            .index
        )

        for account_id, month in problematic_keys:
            print(
                combined.loc[
                    (
                        combined[
                            "contract_account_hashed"
                        ]
                        == account_id
                    )
                    & (
                        combined[
                            "month_year"
                        ]
                        == month
                    )
                ].to_string()
            )

        raise ValueError(
            "Some duplicate postpaid account-months differ "
            "on variables used in the regression."
        )

    duplicates_to_drop = combined.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep="first",
    ).sum()

    print(
        "\nDropping one copy of analytically identical "
        "duplicate account-month observations:",
        f"{duplicates_to_drop:,}"
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
# ESTIMATE POSTPAID BINARY SHS MODEL
###########################################################################

print("\n" + "=" * 90)
print("ESTIMATING POSTPAID FIXED-EFFECTS MODEL")
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

# Cluster standard errors by municipal load-shedding block.
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


###########################################################################
# INTERPRET POSTPAID SHS EFFECT
###########################################################################

print("\n" + "=" * 90)
print("POSTPAID SHS EFFECT")
print("=" * 90)


def coefficient_summary(
    result,
    coefficient_name,
):
    estimate = float(
        result.params[
            coefficient_name
        ]
    )

    standard_error = float(
        np.sqrt(
            result.cov.loc[
                coefficient_name,
                coefficient_name,
            ]
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

    return {
        "coefficient": estimate,
        "standard_error": standard_error,
        "percent_effect": percent_effect,
        "lower_95_percent": lower_percent,
        "upper_95_percent": upper_percent,
    }


postpaid_shs_effect = coefficient_summary(
    results,
    "shs_label_numeric",
)

print(
    "coef="
    f"{postpaid_shs_effect['coefficient']:.4f}, "
    "SE="
    f"{postpaid_shs_effect['standard_error']:.4f}, "
    "effect="
    f"{postpaid_shs_effect['percent_effect']:.2f}%, "
    "95% CI=["
    f"{postpaid_shs_effect['lower_95_percent']:.2f}%, "
    f"{postpaid_shs_effect['upper_95_percent']:.2f}%]"
)



###########################################################################
# OUTAGE EFFECTS FOR NON-PV AND PV HOUSEHOLDS
###########################################################################

def linear_combination(result, terms):
    """
    Compute a linear combination of regression coefficients using the full
    clustered covariance matrix.
    """
    weights = pd.Series(
        0.0,
        index=result.params.index,
    )

    for term, weight in terms.items():
        weights.loc[term] = weight

    estimate = float(
        weights @ result.params
    )

    variance = float(
        weights @ result.cov @ weights
    )

    standard_error = np.sqrt(
        variance
    )

    return estimate, standard_error


def report_linear_effect(
    result,
    terms,
    label,
):
    """
    Report a linear combination with:
      - coefficient
      - clustered standard error
      - two-sided p-value
      - exact percentage effect for a log outcome
      - transformed 95% confidence interval
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
        f"95% CI=[{lower_percent:.3f}%, "
        f"{upper_percent:.3f}%]"
    )

    return {
        "effect": label,
        "coefficient": estimate,
        "standard_error": standard_error,
        "p_value": p_value,
        "percent_effect": percent_effect,
        "lower_95_percent": lower_percent,
        "upper_95_percent": upper_percent,
    }


print("\n" + "=" * 90)
print("POSTPAID OUTAGE EFFECTS BY PV STATUS")
print("=" * 90)

# Effect of one additional Stage-1-equivalent outage unit (2.5 hours)
# for postpaid households without rooftop PV.
non_pv_outage_effect = report_linear_effect(
    results,
    {
        "total_duration_stage1": 1,
    },
    "Non-PV outage effect",
)

# Total outage effect for postpaid households with rooftop PV:
# beta_outage + beta_outage_x_PV.
pv_outage_effect = report_linear_effect(
    results,
    {
        "total_duration_stage1": 1,
        "outage_x_shs": 1,
    },
    "PV outage effect",
)

# Differential outage response associated with rooftop PV.
pv_outage_difference = report_linear_effect(
    results,
    {
        "outage_x_shs": 1,
    },
    "PV x outage differential",
)


###########################################################################
# SAVE RESULTS
###########################################################################

print("\n" + "=" * 90)
print("SAVING OUTPUTS")
print("=" * 90)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

effect_output = pd.DataFrame(
    [
        {
            "sample": "Postpaid",
            **postpaid_shs_effect,
            "observations": int(
                results.nobs
            ),
            "households": int(
                combined.index
                .get_level_values(
                    "contract_account_hashed"
                )
                .nunique()
            ),
            "time_periods": int(
                combined.index
                .get_level_values(
                    "month_year"
                )
                .nunique()
            ),
            "cluster_variable": "Area_number",
            "entity_fixed_effects": True,
            "time_fixed_effects": True,
        }
    ]
)

effect_csv = (
    OUTPUT_DIRECTORY
    / "postpaid_shs_effect_09182026.csv"
)

summary_txt = (
    OUTPUT_DIRECTORY
    / "postpaid_shs_regression_summary_09182026.txt"
)

effect_output.to_csv(
    effect_csv,
    index=False,
)


outage_effects_output = pd.DataFrame(
    [
        non_pv_outage_effect,
        pv_outage_effect,
        pv_outage_difference,
    ]
)

outage_effects_csv = (
    OUTPUT_DIRECTORY
    / "postpaid_outage_effects_by_pv_09182026.csv"
)

outage_effects_output.to_csv(
    outage_effects_csv,
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
    "Saved postpaid SHS effect to:"
)

print(
    effect_csv
)


print(
    "\nSaved postpaid outage-effect linear combinations to:"
)

print(
    outage_effects_csv
)

print(
    "\nSaved postpaid regression summary to:"
)

print(
    summary_txt
)

print(
    "\nProcessing complete!"
)
