import numpy as np
import pandas as pd
from pathlib import Path


###############################################################################
# CONFIG
###############################################################################

PREPAID_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/prepaid_cleaned_vector_noagg.parquet"
)

POSTPAID_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/postpaid_cleaned_vector_noagg.parquet"
)


REGISTRATION_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "5c_out_08052026/with_sseg_reg.parquet"
)

# Residential AMI charges, VAT inclusive, consistent with the tariff values
# used elsewhere in this script.
AMI_FEE_RAND_PER_MONTH = {
    "2022/23": 96.20,
    "2023/24": 5.66,
}


# Counterfactual sensitivity: City retains the Jan-Jun 2023 residential
# AMI fee for the entire calendar year rather than reducing it in Jul-Dec.
HIGH_AMI_FEE_ALL_YEAR = 96.20

OUTPUT_DIR = Path(
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "10_Revenue_09072026_FBEpreferred"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_START = pd.Timestamp("2023-01-01")
ANALYSIS_END = pd.Timestamp("2023-12-31")
HISTORY_START = pd.Timestamp("2022-01-01")

FBE_WINDOW_MONTHS = 12


###############################################################################
# SHS COEFFICIENTS
###############################################################################

SHS_BETA = {
    "LifeLine": 0.003846373360780577,
    "Domestic": 0.00977305697837436,
    "Home User": -0.13278742953647324,
    "Postpaid Home User": -0.1687,
}

print("\nPV coefficients used in revenue analysis:")
for tariff, beta in SHS_BETA.items():
    pct = 100 * (np.exp(beta) - 1)
    print(
        f"{tariff}: beta={beta:.7f}, "
        f"implied grid-purchase effect={pct:.2f}%"
    )

###############################################################################
# 2023 CITY OF CAPE TOWN RESIDENTIAL TARIFFS
###############################################################################

TARIFFS = {
    "2022/23": {
        "LifeLine": {
            "threshold_kwh": 350.0,
            "rate1_rand_per_kwh": 1.7989,
            "rate2_rand_per_kwh": 3.6272,
            "fixed_rand_per_month": 0.0,
        },
        "Domestic": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 2.9830,
            "rate2_rand_per_kwh": 3.6272,
            "fixed_rand_per_month": 0.0,
        },
        "Home User": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 2.6284,
            "rate2_rand_per_kwh": 3.6272,
            "fixed_rand_per_month": 212.75,
        },
        "Postpaid Home User": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 2.6284,
            "rate2_rand_per_kwh": 3.6272,
            "fixed_rand_per_month": 212.75,
        },
    },
    "2023/24": {
        "LifeLine": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 2.1155,
            "rate2_rand_per_kwh": 4.2656,
            "fixed_rand_per_month": 0.0,
        },
        "Domestic": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 3.5080,
            "rate2_rand_per_kwh": 4.2656,
            "fixed_rand_per_month": 0.0,
        },
        "Home User": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 3.0858,
            "rate2_rand_per_kwh": 4.2656,
            "fixed_rand_per_month": 252.09,
        },
        "Postpaid Home User": {
            "threshold_kwh": 600.0,
            "rate1_rand_per_kwh": 3.0858,
            "rate2_rand_per_kwh": 4.2656,
            "fixed_rand_per_month": 252.09,
        },
    },
}


###############################################################################
# HELPERS
###############################################################################

def section(title):
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


def financial_year_for_month(month_year):
    return "2022/23" if month_year.month <= 6 else "2023/24"


def calculate_energy_charge(kwh, tariff_name, financial_year, free_kwh=None):
    """
    Calculate variable electricity revenue.

    FBE is removed from the first tariff block, but the tariff threshold itself
    remains based on total monthly consumption.
    """
    kwh = np.asarray(kwh, dtype=float)
    tariff_name = np.asarray(tariff_name, dtype=object)
    financial_year = np.asarray(financial_year, dtype=object)

    if free_kwh is None:
        free_kwh = np.zeros(len(kwh), dtype=float)
    else:
        free_kwh = np.asarray(free_kwh, dtype=float)

    revenue = np.full(len(kwh), np.nan, dtype=float)

    for fy, fy_tariffs in TARIFFS.items():
        for tariff, config in fy_tariffs.items():
            mask = (
                (financial_year == fy)
                & (tariff_name == tariff)
                & np.isfinite(kwh)
                & np.isfinite(free_kwh)
            )

            if not mask.any():
                continue

            threshold = config["threshold_kwh"]
            rate1 = config["rate1_rand_per_kwh"]
            rate2 = config["rate2_rand_per_kwh"]

            total_first_block = np.minimum(kwh[mask], threshold)

            billed_first_block = np.maximum(
                total_first_block - free_kwh[mask],
                0.0,
            )

            billed_second_block = np.maximum(
                kwh[mask] - threshold,
                0.0,
            )

            revenue[mask] = (
                billed_first_block * rate1
                + billed_second_block * rate2
            )

    return revenue


def calculate_fixed_charge(tariff_name, financial_year):
    tariff_name = np.asarray(tariff_name, dtype=object)
    financial_year = np.asarray(financial_year, dtype=object)

    fixed = np.full(len(tariff_name), np.nan, dtype=float)

    for fy, fy_tariffs in TARIFFS.items():
        for tariff, config in fy_tariffs.items():
            mask = (
                (financial_year == fy)
                & (tariff_name == tariff)
            )

            if mask.any():
                fixed[mask] = config["fixed_rand_per_month"]

    return fixed


def add_total_row(summary):
    total = {
        "tariff_group": "TOTAL",
        "households": np.nan,
        "household_months": summary["household_months"].sum(),
        "baseline_kwh": summary["baseline_kwh"].sum(),
        "scenario_kwh": summary["scenario_kwh"].sum(),
        "kwh_reduction": summary["kwh_reduction"].sum(),
        "baseline_total_revenue_gross": summary[
            "baseline_total_revenue_gross"
        ].sum(),
        "scenario_total_revenue_gross": summary[
            "scenario_total_revenue_gross"
        ].sum(),
        "revenue_loss_gross": summary["revenue_loss_gross"].sum(),
        "baseline_total_revenue_preferred": summary[
            "baseline_total_revenue_preferred"
        ].sum(),
        "scenario_total_revenue_preferred": summary[
            "scenario_total_revenue_preferred"
        ].sum(),
        "revenue_loss_preferred": summary["revenue_loss_preferred"].sum(),
        "lifeline_rows_exact_fbe": summary[
            "lifeline_rows_exact_fbe"
        ].sum(min_count=1),
        "lifeline_rows_gross_fallback": summary[
            "lifeline_rows_gross_fallback"
        ].sum(min_count=1),
    }

    total["pct_kwh_reduction"] = (
        100 * total["kwh_reduction"] / total["baseline_kwh"]
    )

    total["pct_total_revenue_loss_gross"] = (
        100
        * total["revenue_loss_gross"]
        / total["baseline_total_revenue_gross"]
    )

    total["pct_total_revenue_loss_preferred"] = (
        100
        * total["revenue_loss_preferred"]
        / total["baseline_total_revenue_preferred"]
    )

    return pd.concat(
        [summary, pd.DataFrame([total])],
        ignore_index=True,
    )


###############################################################################
# LOAD DATA
###############################################################################

section("LOADING CORRECTED STEP 8 DATA")

columns = [
    "contract_account_hashed",
    "month_year",
    "kwh",
    "tariff_category",
    "shs_label_numeric",
]

prepaid = pd.read_parquet(PREPAID_FILE, columns=columns)
postpaid = pd.read_parquet(POSTPAID_FILE, columns=columns)

prepaid["customer_type"] = "prepaid"
postpaid["customer_type"] = "postpaid"

data = pd.concat(
    [prepaid, postpaid],
    ignore_index=True,
)

data["month_year"] = pd.to_datetime(
    data["month_year"],
    errors="coerce",
)

data["kwh"] = pd.to_numeric(
    data["kwh"],
    errors="coerce",
)

data["shs_label_numeric"] = pd.to_numeric(
    data["shs_label_numeric"],
    errors="coerce",
)

data = data.loc[
    data["month_year"].between(
        HISTORY_START,
        ANALYSIS_END,
        inclusive="both",
    )
].copy()

data = data.loc[
    data["kwh"].notna()
    & data["kwh"].ge(0)
].copy()

duplicate_rows = int(
    data.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Rows involved in duplicate account-months:",
    f"{duplicate_rows:,}",
    flush=True,
)

if duplicate_rows > 0:
    raise ValueError(
        "Revenue input is not unique by household-month."
    )

print(
    "Rows loaded for 2022-2023 history:",
    f"{len(data):,}",
    flush=True,
)

print(
    "Unique households:",
    f"{data['contract_account_hashed'].nunique():,}",
    flush=True,
)


###############################################################################
# DEFINE TARIFF GROUPS
###############################################################################

section("DEFINING TARIFF GROUPS")

data["tariff_group"] = np.where(
    data["customer_type"].eq("postpaid"),
    "Postpaid Home User",
    data["tariff_category"],
)

data = data.loc[
    data["tariff_group"].isin(
        SHS_BETA.keys()
    )
].copy()

print(
    data["tariff_group"].value_counts(
        dropna=False
    ),
    flush=True,
)


###############################################################################
# CONSTRUCT LIFELINE FBE ELIGIBILITY
###############################################################################

section("CONSTRUCTING LIFELINE FBE ELIGIBILITY")

data = data.sort_values(
    [
        "contract_account_hashed",
        "month_year",
    ]
).reset_index(drop=True)

data["prior_12m_avg_kwh"] = (
    data.groupby(
        "contract_account_hashed"
    )["kwh"]
    .transform(
        lambda s: (
            s.shift(1)
            .rolling(
                window=FBE_WINDOW_MONTHS,
                min_periods=FBE_WINDOW_MONTHS,
            )
            .mean()
        )
    )
)

lifeline = data["tariff_group"].eq("LifeLine")

data["fbe_kwh"] = 0.0
data.loc[lifeline, "fbe_kwh"] = np.nan

data.loc[
    lifeline
    & data["prior_12m_avg_kwh"].le(250),
    "fbe_kwh",
] = 60.0

data.loc[
    lifeline
    & data["prior_12m_avg_kwh"].gt(250)
    & data["prior_12m_avg_kwh"].le(450),
    "fbe_kwh",
] = 25.0

data.loc[
    lifeline
    & data["prior_12m_avg_kwh"].gt(450),
    "fbe_kwh",
] = 0.0

data["fbe_history_complete"] = (
    ~lifeline
    | data["fbe_kwh"].notna()
)

print(
    "All-history Lifeline FBE allocation:",
    flush=True,
)

print(
    data.loc[
        lifeline,
        "fbe_kwh",
    ].value_counts(
        dropna=False
    ).sort_index(),
    flush=True,
)


###############################################################################
# RESTRICT TO 2023
###############################################################################

section("RESTRICTING TO 2023")

analysis = data.loc[
    data["month_year"].between(
        ANALYSIS_START,
        ANALYSIS_END,
        inclusive="both",
    )
].copy()

analysis["financial_year"] = (
    analysis["month_year"]
    .apply(financial_year_for_month)
)

analysis["beta"] = (
    analysis["tariff_group"]
    .map(SHS_BETA)
)

analysis["shs_factor"] = np.exp(
    analysis["beta"]
)

print(
    "2023 observations:",
    f"{len(analysis):,}",
    flush=True,
)

print(
    "2023 unique households:",
    f"{analysis['contract_account_hashed'].nunique():,}",
    flush=True,
)

print(
    "\n2023 observations by tariff:",
    flush=True,
)

print(
    analysis["tariff_group"].value_counts(),
    flush=True,
)

print(
    "\n2023 SHS households by tariff:",
    flush=True,
)

print(
    analysis.loc[
        analysis["shs_label_numeric"].eq(1)
    ]
    .groupby("tariff_group")[
        "contract_account_hashed"
    ]
    .nunique(),
    flush=True,
)



###############################################################################
# MERGE OFFICIAL SSEG REGISTRATION STATUS
###############################################################################

section("MERGING OFFICIAL SSEG REGISTRATION STATUS")

registration_columns = [
    "contract_account_hashed",
    "month_year",
    "has_registration",
]

registration = pd.read_parquet(
    REGISTRATION_FILE,
    columns=registration_columns,
)

registration["month_year"] = pd.to_datetime(
    registration["month_year"],
    errors="coerce",
)

registration["has_registration"] = pd.to_numeric(
    registration["has_registration"],
    errors="coerce",
).fillna(0).astype(int)

registration = registration.loc[
    registration["month_year"].between(
        ANALYSIS_START,
        ANALYSIS_END,
        inclusive="both",
    )
].copy()

registration_duplicate_rows = int(
    registration.duplicated(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep=False,
    ).sum()
)

print(
    "Registration rows involved in duplicate account-months:",
    f"{registration_duplicate_rows:,}",
    flush=True,
)

if registration_duplicate_rows > 0:
    raise ValueError(
        "Registration input is not unique by household-month."
    )

rows_before_registration_merge = len(analysis)

analysis = analysis.merge(
    registration,
    on=[
        "contract_account_hashed",
        "month_year",
    ],
    how="left",
    validate="one_to_one",
)

if len(analysis) != rows_before_registration_merge:
    raise ValueError(
        "Registration merge changed the number of revenue-analysis rows."
    )

analysis["has_registration"] = (
    analysis["has_registration"]
    .fillna(0)
    .astype(int)
)

pv_2023 = analysis["shs_label_numeric"].eq(1)
registered_pv_2023 = (
    pv_2023
    & analysis["has_registration"].eq(1)
)

print(
    "PV household-months in 2023:",
    f"{pv_2023.sum():,}",
    flush=True,
)

print(
    "Registered PV household-months in 2023:",
    f"{registered_pv_2023.sum():,}",
    flush=True,
)

print(
    "Unique PV households in 2023:",
    f"{analysis.loc[pv_2023, 'contract_account_hashed'].nunique():,}",
    flush=True,
)

print(
    "Unique PV households registered during a PV month in 2023:",
    f"{analysis.loc[registered_pv_2023, 'contract_account_hashed'].nunique():,}",
    flush=True,
)

print(
    "\nRegistered PV households by tariff:",
    flush=True,
)

print(
    analysis.loc[
        registered_pv_2023
    ].groupby(
        "tariff_group"
    )["contract_account_hashed"].nunique(),
    flush=True,
)


###############################################################################
# LIFELINE FBE QA
###############################################################################

section("2023 LIFELINE FBE QA")

lifeline_2023 = analysis[
    "tariff_group"
].eq("LifeLine")

complete_lifeline_2023 = (
    lifeline_2023
    & analysis["fbe_history_complete"]
)

incomplete_lifeline_2023 = (
    lifeline_2023
    & ~analysis["fbe_history_complete"]
)

print(
    "2023 Lifeline observations:",
    f"{lifeline_2023.sum():,}",
    flush=True,
)

print(
    "2023 Lifeline rows with exact FBE eligibility:",
    f"{complete_lifeline_2023.sum():,}",
    flush=True,
)

print(
    "2023 Lifeline rows using gross-billing fallback:",
    f"{incomplete_lifeline_2023.sum():,}",
    flush=True,
)

print(
    "Share with exact FBE eligibility:",
    f"{100 * complete_lifeline_2023.sum() / lifeline_2023.sum():.2f}%",
    flush=True,
)

print(
    "\nFBE allocation among exact-history Lifeline rows:",
    flush=True,
)

print(
    analysis.loc[
        complete_lifeline_2023,
        "fbe_kwh",
    ].value_counts().sort_index(),
    flush=True,
)


###############################################################################
# CONSTRUCT COUNTERFACTUAL CONSUMPTION
###############################################################################

section("CONSTRUCTING CONSUMPTION COUNTERFACTUALS")

existing_shs = analysis[
    "shs_label_numeric"
].eq(1)

analysis["baseline_no_shs_kwh"] = analysis[
    "kwh"
].copy()

analysis.loc[
    existing_shs,
    "baseline_no_shs_kwh",
] = (
    analysis.loc[
        existing_shs,
        "kwh",
    ]
    / analysis.loc[
        existing_shs,
        "shs_factor",
    ]
)

analysis["actual_adoption_kwh"] = analysis[
    "baseline_no_shs_kwh"
].copy()

analysis.loc[
    existing_shs,
    "actual_adoption_kwh",
] = analysis.loc[
    existing_shs,
    "kwh",
]

analysis["universal_shs_kwh"] = (
    analysis["baseline_no_shs_kwh"]
    * analysis["shs_factor"]
)

print(
    "Existing SHS household-months:",
    f"{existing_shs.sum():,}",
    flush=True,
)

print(
    "No-SHS baseline kWh:",
    f"{analysis['baseline_no_shs_kwh'].sum():,.0f}",
    flush=True,
)

print(
    "Actual-adoption kWh:",
    f"{analysis['actual_adoption_kwh'].sum():,.0f}",
    flush=True,
)

print(
    "Universal-adoption kWh:",
    f"{analysis['universal_shs_kwh'].sum():,.0f}",
    flush=True,
)


###############################################################################
# SCENARIO SUMMARY
###############################################################################

def summarize_scenario(
    frame,
    scenario_name,
    scenario_kwh_col,
):
    """
    Preferred revenue:
      - Non-Lifeline: standard billing.
      - Lifeline with full prior 12-month history: exact FBE-adjusted billing.
      - Lifeline without full history: gross-of-FBE billing fallback.

    Gross revenue:
      - ignores FBE for all rows and is retained as a sensitivity check.
    """
    tmp = frame.copy()

    zero_fbe = np.zeros(
        len(tmp),
        dtype=float,
    )

    baseline_energy_gross = calculate_energy_charge(
        tmp["baseline_no_shs_kwh"],
        tmp["tariff_group"],
        tmp["financial_year"],
        free_kwh=zero_fbe,
    )

    scenario_energy_gross = calculate_energy_charge(
        tmp[scenario_kwh_col],
        tmp["tariff_group"],
        tmp["financial_year"],
        free_kwh=zero_fbe,
    )

    fixed_revenue = calculate_fixed_charge(
        tmp["tariff_group"],
        tmp["financial_year"],
    )

    tmp["baseline_total_revenue_gross"] = (
        baseline_energy_gross
        + fixed_revenue
    )

    tmp["scenario_total_revenue_gross"] = (
        scenario_energy_gross
        + fixed_revenue
    )

    tmp["revenue_loss_gross"] = (
        tmp["baseline_total_revenue_gross"]
        - tmp["scenario_total_revenue_gross"]
    )

    # Preferred starts from full-sample gross billing.
    tmp["baseline_total_revenue_preferred"] = (
        tmp["baseline_total_revenue_gross"].copy()
    )

    tmp["scenario_total_revenue_preferred"] = (
        tmp["scenario_total_revenue_gross"].copy()
    )

    tmp["revenue_loss_preferred"] = (
        tmp["revenue_loss_gross"].copy()
    )

    # Overwrite exact-history Lifeline rows with FBE-adjusted billing.
    lifeline_exact = (
        tmp["tariff_group"].eq("LifeLine")
        & tmp["fbe_history_complete"]
    )

    exact_subset = tmp.loc[
        lifeline_exact
    ].copy()

    if not exact_subset.empty:

        baseline_exact_energy = calculate_energy_charge(
            exact_subset[
                "baseline_no_shs_kwh"
            ],
            exact_subset[
                "tariff_group"
            ],
            exact_subset[
                "financial_year"
            ],
            free_kwh=exact_subset[
                "fbe_kwh"
            ],
        )

        scenario_exact_energy = calculate_energy_charge(
            exact_subset[
                scenario_kwh_col
            ],
            exact_subset[
                "tariff_group"
            ],
            exact_subset[
                "financial_year"
            ],
            free_kwh=exact_subset[
                "fbe_kwh"
            ],
        )

        fixed_exact = calculate_fixed_charge(
            exact_subset[
                "tariff_group"
            ],
            exact_subset[
                "financial_year"
            ],
        )

        baseline_exact_total = (
            baseline_exact_energy
            + fixed_exact
        )

        scenario_exact_total = (
            scenario_exact_energy
            + fixed_exact
        )

        tmp.loc[
            exact_subset.index,
            "baseline_total_revenue_preferred",
        ] = baseline_exact_total

        tmp.loc[
            exact_subset.index,
            "scenario_total_revenue_preferred",
        ] = scenario_exact_total

        tmp.loc[
            exact_subset.index,
            "revenue_loss_preferred",
        ] = (
            baseline_exact_total
            - scenario_exact_total
        )

    tmp["kwh_reduction"] = (
        tmp["baseline_no_shs_kwh"]
        - tmp[scenario_kwh_col]
    )

    rows = []

    for tariff_group, group in tmp.groupby(
        "tariff_group",
        observed=True,
    ):
        baseline_gross = group[
            "baseline_total_revenue_gross"
        ].sum()

        scenario_gross = group[
            "scenario_total_revenue_gross"
        ].sum()

        loss_gross = group[
            "revenue_loss_gross"
        ].sum()

        baseline_preferred = group[
            "baseline_total_revenue_preferred"
        ].sum()

        scenario_preferred = group[
            "scenario_total_revenue_preferred"
        ].sum()

        loss_preferred = group[
            "revenue_loss_preferred"
        ].sum()

        row = {
            "tariff_group": tariff_group,
            "household_months": len(group),
            "households": group[
                "contract_account_hashed"
            ].nunique(),
            "baseline_kwh": group[
                "baseline_no_shs_kwh"
            ].sum(),
            "scenario_kwh": group[
                scenario_kwh_col
            ].sum(),
            "kwh_reduction": group[
                "kwh_reduction"
            ].sum(),
            "baseline_total_revenue_gross": baseline_gross,
            "scenario_total_revenue_gross": scenario_gross,
            "revenue_loss_gross": loss_gross,
            "baseline_total_revenue_preferred": baseline_preferred,
            "scenario_total_revenue_preferred": scenario_preferred,
            "revenue_loss_preferred": loss_preferred,
            "lifeline_rows_exact_fbe": (
                int(
                    group["fbe_history_complete"].sum()
                )
                if tariff_group == "LifeLine"
                else np.nan
            ),
            "lifeline_rows_gross_fallback": (
                int(
                    (~group["fbe_history_complete"]).sum()
                )
                if tariff_group == "LifeLine"
                else np.nan
            ),
        }

        row["pct_kwh_reduction"] = (
            100
            * row["kwh_reduction"]
            / row["baseline_kwh"]
        )

        row["pct_total_revenue_loss_gross"] = (
            100
            * loss_gross
            / baseline_gross
        )

        row["pct_total_revenue_loss_preferred"] = (
            100
            * loss_preferred
            / baseline_preferred
        )

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["scenario"] = scenario_name

    return add_total_row(summary), tmp



###############################################################################
# HOME USER TARIFF-CLASS COMPLIANCE COUNTERFACTUAL
###############################################################################

def calculate_home_user_tariff_compliance(
    frame,
    actual_detail,
):
    """
    Counterfactual policy scenario:
      - Keep observed 2023 SHS adoption and observed SHS consumption unchanged.
      - For household-months observed with SHS:
          * LifeLine -> Home User tariff
          * Domestic -> Home User tariff
          * Home User -> unchanged
          * Postpaid Home User -> unchanged
      - Non-SHS household-months remain on their observed tariff.
      - Lifeline FBE continues to apply only to household-months that remain
        on the Lifeline tariff.
      - Feed-in tariff payments and AMI administration fees are excluded.

    The main quantity of interest is additional municipal revenue from enforcing
    the Home User tariff requirement among detected PV customers currently
    observed on LifeLine or Domestic tariffs.
    """

    tmp = frame.copy()

    if not tmp.index.equals(actual_detail.index):
        raise ValueError(
            "Analysis frame and actual-detail frame do not have matching indices."
        )

    # Preserve original tariff for grouping/reporting.
    tmp["original_tariff_group"] = tmp["tariff_group"]

    existing_shs = tmp["shs_label_numeric"].eq(1)

    # Only LifeLine and Domestic SHS observations are switched.
    switch_to_home_user = (
        existing_shs
        & tmp["tariff_group"].isin(
            ["LifeLine", "Domestic"]
        )
    )

    tmp["compliance_tariff_group"] = tmp["tariff_group"]

    tmp.loc[
        switch_to_home_user,
        "compliance_tariff_group",
    ] = "Home User"

    # Keep the observed-adoption kWh unchanged. This isolates the billing/tariff
    # effect of compliance rather than changing electricity consumption.
    tmp["compliance_kwh"] = tmp["actual_adoption_kwh"]

    # Gross compliance billing.
    zero_fbe = np.zeros(
        len(tmp),
        dtype=float,
    )

    compliance_energy_gross = calculate_energy_charge(
        tmp["compliance_kwh"],
        tmp["compliance_tariff_group"],
        tmp["financial_year"],
        free_kwh=zero_fbe,
    )

    compliance_fixed = calculate_fixed_charge(
        tmp["compliance_tariff_group"],
        tmp["financial_year"],
    )

    tmp["compliance_total_revenue_gross"] = (
        compliance_energy_gross
        + compliance_fixed
    )

    # Preferred compliance billing starts from gross billing.
    tmp["compliance_total_revenue_preferred"] = (
        tmp["compliance_total_revenue_gross"].copy()
    )

    # Only observations that remain Lifeline are potentially FBE-adjusted.
    lifeline_exact_compliance = (
        tmp["compliance_tariff_group"].eq("LifeLine")
        & tmp["fbe_history_complete"]
    )

    exact_subset = tmp.loc[
        lifeline_exact_compliance
    ].copy()

    if not exact_subset.empty:

        exact_energy = calculate_energy_charge(
            exact_subset["compliance_kwh"],
            exact_subset["compliance_tariff_group"],
            exact_subset["financial_year"],
            free_kwh=exact_subset["fbe_kwh"],
        )

        exact_fixed = calculate_fixed_charge(
            exact_subset["compliance_tariff_group"],
            exact_subset["financial_year"],
        )

        tmp.loc[
            exact_subset.index,
            "compliance_total_revenue_preferred",
        ] = (
            exact_energy
            + exact_fixed
        )

    # Observed-adoption revenue from the already-computed main scenario.
    tmp["observed_adoption_revenue_preferred"] = (
        actual_detail[
            "scenario_total_revenue_preferred"
        ].to_numpy()
    )

    tmp["observed_adoption_revenue_gross"] = (
        actual_detail[
            "scenario_total_revenue_gross"
        ].to_numpy()
    )

    # No-SHS baseline revenue from the main scenario, used to calculate
    # remaining revenue loss after compliance.
    tmp["baseline_no_shs_revenue_preferred"] = (
        actual_detail[
            "baseline_total_revenue_preferred"
        ].to_numpy()
    )

    tmp["baseline_no_shs_revenue_gross"] = (
        actual_detail[
            "baseline_total_revenue_gross"
        ].to_numpy()
    )

    # Revenue recovered through tariff compliance.
    tmp["revenue_recovered_preferred"] = (
        tmp["compliance_total_revenue_preferred"]
        - tmp["observed_adoption_revenue_preferred"]
    )

    tmp["revenue_recovered_gross"] = (
        tmp["compliance_total_revenue_gross"]
        - tmp["observed_adoption_revenue_gross"]
    )

    # Remaining PV-related revenue loss after tariff compliance.
    tmp["remaining_revenue_loss_preferred"] = (
        tmp["baseline_no_shs_revenue_preferred"]
        - tmp["compliance_total_revenue_preferred"]
    )

    tmp["remaining_revenue_loss_gross"] = (
        tmp["baseline_no_shs_revenue_gross"]
        - tmp["compliance_total_revenue_gross"]
    )

    # Flag whether the household-month actually changes tariff.
    tmp["switched_to_home_user"] = switch_to_home_user.astype(int)

    rows = []

    for tariff_group, group in tmp.groupby(
        "original_tariff_group",
        observed=True,
    ):

        original_loss_preferred = (
            group["baseline_no_shs_revenue_preferred"].sum()
            - group["observed_adoption_revenue_preferred"].sum()
        )

        recovered_preferred = group[
            "revenue_recovered_preferred"
        ].sum()

        remaining_loss_preferred = group[
            "remaining_revenue_loss_preferred"
        ].sum()

        original_loss_gross = (
            group["baseline_no_shs_revenue_gross"].sum()
            - group["observed_adoption_revenue_gross"].sum()
        )

        recovered_gross = group[
            "revenue_recovered_gross"
        ].sum()

        remaining_loss_gross = group[
            "remaining_revenue_loss_gross"
        ].sum()

        pv_mask = group["shs_label_numeric"].eq(1)

        rows.append(
            {
                "original_tariff_group": tariff_group,
                "households": group[
                    "contract_account_hashed"
                ].nunique(),
                "pv_households": group.loc[
                    pv_mask,
                    "contract_account_hashed",
                ].nunique(),
                "pv_household_months": int(
                    pv_mask.sum()
                ),
                "switched_household_months": int(
                    group["switched_to_home_user"].sum()
                ),
                "switched_households": group.loc[
                    group["switched_to_home_user"].eq(1),
                    "contract_account_hashed",
                ].nunique(),
                "observed_revenue_loss_preferred": (
                    original_loss_preferred
                ),
                "revenue_recovered_preferred": (
                    recovered_preferred
                ),
                "remaining_revenue_loss_preferred": (
                    remaining_loss_preferred
                ),
                "pct_original_loss_recovered_preferred": (
                    100
                    * recovered_preferred
                    / original_loss_preferred
                    if original_loss_preferred != 0
                    else np.nan
                ),
                "observed_revenue_loss_gross": (
                    original_loss_gross
                ),
                "revenue_recovered_gross": (
                    recovered_gross
                ),
                "remaining_revenue_loss_gross": (
                    remaining_loss_gross
                ),
                "pct_original_loss_recovered_gross": (
                    100
                    * recovered_gross
                    / original_loss_gross
                    if original_loss_gross != 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)

    total_original_loss_preferred = summary[
        "observed_revenue_loss_preferred"
    ].sum()

    total_recovered_preferred = summary[
        "revenue_recovered_preferred"
    ].sum()

    total_remaining_preferred = summary[
        "remaining_revenue_loss_preferred"
    ].sum()

    total_original_loss_gross = summary[
        "observed_revenue_loss_gross"
    ].sum()

    total_recovered_gross = summary[
        "revenue_recovered_gross"
    ].sum()

    total_remaining_gross = summary[
        "remaining_revenue_loss_gross"
    ].sum()

    total_row = pd.DataFrame(
        [
            {
                "original_tariff_group": "TOTAL",
                "households": np.nan,
                "pv_households": np.nan,
                "pv_household_months": int(
                    tmp["shs_label_numeric"].eq(1).sum()
                ),
                "switched_household_months": int(
                    tmp["switched_to_home_user"].sum()
                ),
                "switched_households": tmp.loc[
                    tmp["switched_to_home_user"].eq(1),
                    "contract_account_hashed",
                ].nunique(),
                "observed_revenue_loss_preferred": (
                    total_original_loss_preferred
                ),
                "revenue_recovered_preferred": (
                    total_recovered_preferred
                ),
                "remaining_revenue_loss_preferred": (
                    total_remaining_preferred
                ),
                "pct_original_loss_recovered_preferred": (
                    100
                    * total_recovered_preferred
                    / total_original_loss_preferred
                    if total_original_loss_preferred != 0
                    else np.nan
                ),
                "observed_revenue_loss_gross": (
                    total_original_loss_gross
                ),
                "revenue_recovered_gross": (
                    total_recovered_gross
                ),
                "remaining_revenue_loss_gross": (
                    total_remaining_gross
                ),
                "pct_original_loss_recovered_gross": (
                    100
                    * total_recovered_gross
                    / total_original_loss_gross
                    if total_original_loss_gross != 0
                    else np.nan
                ),
            }
        ]
    )

    summary = pd.concat(
        [summary, total_row],
        ignore_index=True,
    )

    return summary, tmp



###############################################################################
# FULL SSEG REGISTRATION / AMI COMPLIANCE COUNTERFACTUAL
###############################################################################

def calculate_full_registration_ami_compliance(
    frame,
    actual_detail,
    tariff_compliance_detail,
):
    """
    Full SSEG registration/AMI compliance counterfactual.

    This builds on the Home User tariff-class compliance counterfactual:
      1. LifeLine and Domestic PV household-months are billed as Home User.
      2. All detected PV household-months are assumed formally registered.
      3. Registered residential SSEG customers are charged the applicable
         monthly residential AMI fee.
      4. The incremental AMI-revenue calculation counts only PV household-
         months that are not already registered in the observed data, because
         already-registered customers should already be paying the fee.
      5. Electricity consumption is held fixed at observed-adoption kWh.
      6. Feed-in tariff payments are excluded.

    The output therefore separates:
      - revenue recovered from Home User tariff-class compliance;
      - additional AMI revenue from registering currently unregistered PV;
      - total additional revenue under combined compliance.
    """

    tmp = frame.copy()

    if not tmp.index.equals(actual_detail.index):
        raise ValueError(
            "Analysis frame and actual-detail frame do not have matching indices."
        )

    if not tmp.index.equals(tariff_compliance_detail.index):
        raise ValueError(
            "Analysis frame and tariff-compliance frame do not have matching indices."
        )

    tmp["original_tariff_group"] = tmp["tariff_group"]

    pv_mask = tmp["shs_label_numeric"].eq(1)
    registered_pv_mask = (
        pv_mask
        & tmp["has_registration"].eq(1)
    )
    unregistered_pv_mask = (
        pv_mask
        & tmp["has_registration"].eq(0)
    )

    # Revenue under Home User tariff-class compliance, before AMI.
    tmp["tariff_compliance_revenue_preferred"] = (
        tariff_compliance_detail[
            "compliance_total_revenue_preferred"
        ].to_numpy()
    )

    tmp["observed_adoption_revenue_preferred"] = (
        actual_detail[
            "scenario_total_revenue_preferred"
        ].to_numpy()
    )

    tmp["baseline_no_shs_revenue_preferred"] = (
        actual_detail[
            "baseline_total_revenue_preferred"
        ].to_numpy()
    )

    # Monthly AMI fee by financial year.
    tmp["ami_fee_rand"] = (
        tmp["financial_year"]
        .map(AMI_FEE_RAND_PER_MONTH)
        .astype(float)
    )

    # Already-registered PV AMI fees are NOT counted as newly recovered revenue;
    # they are reported separately as existing registration-related revenue.
    tmp["observed_registered_ami_revenue"] = np.where(
        registered_pv_mask,
        tmp["ami_fee_rand"],
        0.0,
    )

    # Full-compliance AMI revenue if every detected PV household-month
    # were registered.
    tmp["full_compliance_ami_revenue"] = np.where(
        pv_mask,
        tmp["ami_fee_rand"],
        0.0,
    )

    # Incremental AMI revenue obtained by registering PV household-months
    # not already registered in the observed data.
    tmp["incremental_ami_revenue"] = np.where(
        unregistered_pv_mask,
        tmp["ami_fee_rand"],
        0.0,
    )

    # Revenue recovered through tariff-class compliance.
    tmp["tariff_revenue_recovered"] = (
        tmp["tariff_compliance_revenue_preferred"]
        - tmp["observed_adoption_revenue_preferred"]
    )

    # Combined newly recovered revenue.
    tmp["combined_revenue_recovered"] = (
        tmp["tariff_revenue_recovered"]
        + tmp["incremental_ami_revenue"]
    )

    # Counterfactual municipal revenue under full tariff + registration compliance.
    tmp["full_compliance_revenue_preferred"] = (
        tmp["tariff_compliance_revenue_preferred"]
        + tmp["incremental_ami_revenue"]
    )

    tmp["remaining_revenue_loss_after_full_compliance"] = (
        tmp["baseline_no_shs_revenue_preferred"]
        - tmp["full_compliance_revenue_preferred"]
    )

    rows = []

    for tariff_group, group in tmp.groupby(
        "original_tariff_group",
        observed=True,
    ):

        group_pv = group["shs_label_numeric"].eq(1)
        group_registered_pv = (
            group_pv
            & group["has_registration"].eq(1)
        )
        group_unregistered_pv = (
            group_pv
            & group["has_registration"].eq(0)
        )

        observed_loss = (
            group["baseline_no_shs_revenue_preferred"].sum()
            - group["observed_adoption_revenue_preferred"].sum()
        )

        tariff_recovered = group[
            "tariff_revenue_recovered"
        ].sum()

        existing_ami = group[
            "observed_registered_ami_revenue"
        ].sum()

        incremental_ami = group[
            "incremental_ami_revenue"
        ].sum()

        combined_recovered = group[
            "combined_revenue_recovered"
        ].sum()

        remaining_loss = group[
            "remaining_revenue_loss_after_full_compliance"
        ].sum()

        rows.append(
            {
                "original_tariff_group": tariff_group,
                "pv_households": group.loc[
                    group_pv,
                    "contract_account_hashed",
                ].nunique(),
                "registered_pv_households": group.loc[
                    group_registered_pv,
                    "contract_account_hashed",
                ].nunique(),
                "unregistered_pv_households": group.loc[
                    group_unregistered_pv,
                    "contract_account_hashed",
                ].nunique(),
                "pv_household_months": int(
                    group_pv.sum()
                ),
                "registered_pv_household_months": int(
                    group_registered_pv.sum()
                ),
                "unregistered_pv_household_months": int(
                    group_unregistered_pv.sum()
                ),
                "observed_revenue_loss_preferred": observed_loss,
                "tariff_revenue_recovered": tariff_recovered,
                "existing_registered_ami_revenue": existing_ami,
                "incremental_ami_revenue": incremental_ami,
                "combined_revenue_recovered": combined_recovered,
                "remaining_revenue_loss_after_full_compliance": (
                    remaining_loss
                ),
                "pct_original_loss_recovered_combined": (
                    100
                    * combined_recovered
                    / observed_loss
                    if observed_loss != 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)

    total_observed_loss = summary[
        "observed_revenue_loss_preferred"
    ].sum()

    total_tariff_recovered = summary[
        "tariff_revenue_recovered"
    ].sum()

    total_existing_ami = summary[
        "existing_registered_ami_revenue"
    ].sum()

    total_incremental_ami = summary[
        "incremental_ami_revenue"
    ].sum()

    total_combined_recovered = summary[
        "combined_revenue_recovered"
    ].sum()

    total_remaining_loss = summary[
        "remaining_revenue_loss_after_full_compliance"
    ].sum()

    total_row = pd.DataFrame(
        [
            {
                "original_tariff_group": "TOTAL",
                "pv_households": np.nan,
                "registered_pv_households": np.nan,
                "unregistered_pv_households": np.nan,
                "pv_household_months": int(
                    pv_mask.sum()
                ),
                "registered_pv_household_months": int(
                    registered_pv_mask.sum()
                ),
                "unregistered_pv_household_months": int(
                    unregistered_pv_mask.sum()
                ),
                "observed_revenue_loss_preferred": (
                    total_observed_loss
                ),
                "tariff_revenue_recovered": (
                    total_tariff_recovered
                ),
                "existing_registered_ami_revenue": (
                    total_existing_ami
                ),
                "incremental_ami_revenue": (
                    total_incremental_ami
                ),
                "combined_revenue_recovered": (
                    total_combined_recovered
                ),
                "remaining_revenue_loss_after_full_compliance": (
                    total_remaining_loss
                ),
                "pct_original_loss_recovered_combined": (
                    100
                    * total_combined_recovered
                    / total_observed_loss
                    if total_observed_loss != 0
                    else np.nan
                ),
            }
        ]
    )

    summary = pd.concat(
        [summary, total_row],
        ignore_index=True,
    )

    return summary, tmp


###############################################################################
# ACTUAL ADOPTION
###############################################################################

section("CALCULATING ACTUAL-ADOPTION REVENUE LOSS")

actual_summary, actual_detail = summarize_scenario(
    analysis,
    scenario_name="Actual observed SHS adoption",
    scenario_kwh_col="actual_adoption_kwh",
)

print(
    actual_summary.to_string(
        index=False
    ),
    flush=True,
)


###############################################################################
# UNIVERSAL ADOPTION
###############################################################################

section("CALCULATING UNIVERSAL-ADOPTION REVENUE LOSS")

universal_summary, universal_detail = summarize_scenario(
    analysis,
    scenario_name="Universal SHS adoption",
    scenario_kwh_col="universal_shs_kwh",
)

print(
    universal_summary.to_string(
        index=False
    ),
    flush=True,
)



###############################################################################
# HOME USER TARIFF-CLASS COMPLIANCE
###############################################################################

section("CALCULATING HOME USER TARIFF-CLASS COMPLIANCE COUNTERFACTUAL")

tariff_compliance_summary, tariff_compliance_detail = (
    calculate_home_user_tariff_compliance(
        analysis,
        actual_detail,
    )
)

print(
    tariff_compliance_summary.to_string(
        index=False
    ),
    flush=True,
)

print(
    "\nInterpretation: revenue_recovered_preferred is the additional "
    "municipal revenue that would have been collected in the observed "
    "2023 PV-adoption scenario if detected LifeLine and Domestic PV "
    "household-months had instead been billed on the Home User tariff.",
    flush=True,
)



###############################################################################
# FULL SSEG REGISTRATION / AMI COMPLIANCE
###############################################################################

section("CALCULATING FULL SSEG REGISTRATION / AMI COMPLIANCE")

full_registration_summary, full_registration_detail = (
    calculate_full_registration_ami_compliance(
        analysis,
        actual_detail,
        tariff_compliance_detail,
    )
)

print(
    full_registration_summary.to_string(
        index=False
    ),
    flush=True,
)

print(
    "\nInterpretation: tariff_revenue_recovered is the additional revenue "
    "from moving detected LifeLine/Domestic PV observations onto Home User. "
    "incremental_ami_revenue is the additional AMI revenue from detected PV "
    "household-months not already registered in the official data. "
    "combined_revenue_recovered is the sum of these two policy channels.",
    flush=True,
)


###############################################################################
# TARGETED REGISTRATION / AMI COMPLIANCE: HOME USER + POSTPAID ONLY
###############################################################################

section("CALCULATING TARGETED REGISTRATION / AMI COMPLIANCE: HOME USER + POSTPAID")

targeted_tariffs = [
    "Home User",
    "Postpaid Home User",
]

targeted_detail = full_registration_detail.loc[
    full_registration_detail["original_tariff_group"].isin(
        targeted_tariffs
    )
].copy()

targeted_rows = []

for tariff_group, group in targeted_detail.groupby(
    "original_tariff_group",
    observed=True,
):

    pv_mask_group = group["shs_label_numeric"].eq(1)
    registered_mask_group = (
        pv_mask_group
        & group["has_registration"].eq(1)
    )
    unregistered_mask_group = (
        pv_mask_group
        & group["has_registration"].eq(0)
    )

    observed_loss = (
        group["baseline_no_shs_revenue_preferred"].sum()
        - group["observed_adoption_revenue_preferred"].sum()
    )

    incremental_ami = group[
        "incremental_ami_revenue"
    ].sum()

    targeted_rows.append(
        {
            "tariff_group": tariff_group,
            "pv_households": group.loc[
                pv_mask_group,
                "contract_account_hashed",
            ].nunique(),
            "registered_pv_households": group.loc[
                registered_mask_group,
                "contract_account_hashed",
            ].nunique(),
            "unregistered_pv_households": group.loc[
                unregistered_mask_group,
                "contract_account_hashed",
            ].nunique(),
            "pv_household_months": int(
                pv_mask_group.sum()
            ),
            "unregistered_pv_household_months": int(
                unregistered_mask_group.sum()
            ),
            "observed_pv_revenue_loss": observed_loss,
            "incremental_ami_revenue": incremental_ami,
            "remaining_pv_revenue_loss": (
                observed_loss - incremental_ami
            ),
            "pct_pv_revenue_loss_offset": (
                100 * incremental_ami / observed_loss
                if observed_loss != 0
                else np.nan
            ),
        }
    )

targeted_compliance_summary = pd.DataFrame(
    targeted_rows
)

targeted_observed_loss = targeted_compliance_summary[
    "observed_pv_revenue_loss"
].sum()

targeted_incremental_ami = targeted_compliance_summary[
    "incremental_ami_revenue"
].sum()

targeted_total = pd.DataFrame(
    [
        {
            "tariff_group": "TOTAL",
            "pv_households": targeted_detail.loc[
                targeted_detail["shs_label_numeric"].eq(1),
                "contract_account_hashed",
            ].nunique(),
            "registered_pv_households": targeted_detail.loc[
                targeted_detail["shs_label_numeric"].eq(1)
                & targeted_detail["has_registration"].eq(1),
                "contract_account_hashed",
            ].nunique(),
            "unregistered_pv_households": targeted_detail.loc[
                targeted_detail["shs_label_numeric"].eq(1)
                & targeted_detail["has_registration"].eq(0),
                "contract_account_hashed",
            ].nunique(),
            "pv_household_months": int(
                targeted_detail["shs_label_numeric"].eq(1).sum()
            ),
            "unregistered_pv_household_months": int(
                (
                    targeted_detail["shs_label_numeric"].eq(1)
                    & targeted_detail["has_registration"].eq(0)
                ).sum()
            ),
            "observed_pv_revenue_loss": targeted_observed_loss,
            "incremental_ami_revenue": targeted_incremental_ami,
            "remaining_pv_revenue_loss": (
                targeted_observed_loss
                - targeted_incremental_ami
            ),
            "pct_pv_revenue_loss_offset": (
                100
                * targeted_incremental_ami
                / targeted_observed_loss
                if targeted_observed_loss != 0
                else np.nan
            ),
        }
    ]
)

targeted_compliance_summary = pd.concat(
    [
        targeted_compliance_summary,
        targeted_total,
    ],
    ignore_index=True,
)


# Put targeted AMI compliance revenue in the context of total residential
# municipal electricity revenue in the 2023 analysis sample.
total_residential_baseline_revenue = actual_summary.loc[
    actual_summary["tariff_group"].eq("TOTAL"),
    "baseline_total_revenue_preferred",
].iloc[0]

total_residential_observed_revenue = actual_summary.loc[
    actual_summary["tariff_group"].eq("TOTAL"),
    "scenario_total_revenue_preferred",
].iloc[0]

targeted_total_mask = targeted_compliance_summary["tariff_group"].eq("TOTAL")

targeted_compliance_summary.loc[
    targeted_total_mask,
    "pct_total_residential_baseline_revenue"
] = (
    100
    * targeted_incremental_ami
    / total_residential_baseline_revenue
)

targeted_compliance_summary.loc[
    targeted_total_mask,
    "pct_total_residential_observed_revenue"
] = (
    100
    * targeted_incremental_ami
    / total_residential_observed_revenue
)

# Also populate tariff-specific rows relative to the relevant tariff's
# baseline and observed-adoption revenue.
for idx in targeted_compliance_summary.index[
    ~targeted_total_mask
]:
    tariff = targeted_compliance_summary.loc[idx, "tariff_group"]

    tariff_actual = actual_summary.loc[
        actual_summary["tariff_group"].eq(tariff)
    ]

    if len(tariff_actual) == 1:
        incremental_ami = targeted_compliance_summary.loc[
            idx,
            "incremental_ami_revenue",
        ]

        targeted_compliance_summary.loc[
            idx,
            "pct_tariff_baseline_revenue"
        ] = (
            100
            * incremental_ami
            / tariff_actual[
                "baseline_total_revenue_preferred"
            ].iloc[0]
        )

        targeted_compliance_summary.loc[
            idx,
            "pct_tariff_observed_revenue"
        ] = (
            100
            * incremental_ami
            / tariff_actual[
                "scenario_total_revenue_preferred"
            ].iloc[0]
        )

print(
    targeted_compliance_summary.to_string(
        index=False
    ),
    flush=True,
)

print(
    "\nInterpretation: this targeted scenario leaves Lifeline and Domestic "
    "households completely unchanged and asks only how much additional "
    "residential AMI revenue would be collected if detected PV customers "
    "already on Home User or Postpaid Home User were fully registered.",
    flush=True,
)


targeted_total_result = targeted_compliance_summary.loc[
    targeted_compliance_summary["tariff_group"].eq("TOTAL")
].iloc[0]


###############################################################################
# TARGETED REGISTRATION COMPLIANCE WITH HIGH AMI FEE RETAINED ALL YEAR
###############################################################################

section("CALCULATING TARGETED REGISTRATION WITH HIGH AMI FEE RETAINED ALL YEAR")

high_ami_detail = targeted_detail.copy()

high_ami_pv = high_ami_detail["shs_label_numeric"].eq(1)
high_ami_unregistered = (
    high_ami_pv
    & high_ami_detail["has_registration"].eq(0)
)

# Counterfactual AMI fee: R96.20 in every month of 2023.
high_ami_detail["counterfactual_ami_fee_rand"] = HIGH_AMI_FEE_ALL_YEAR

# Incremental AMI revenue under full registration with the higher fee retained.
high_ami_detail["incremental_ami_revenue_high_fee"] = np.where(
    high_ami_unregistered,
    high_ami_detail["counterfactual_ami_fee_rand"],
    0.0,
)

# Compare against the current-policy targeted AMI scenario.
high_ami_detail["incremental_ami_revenue_current_policy"] = (
    high_ami_detail["incremental_ami_revenue"]
)

high_ami_detail["additional_revenue_from_retaining_high_fee"] = (
    high_ami_detail["incremental_ami_revenue_high_fee"]
    - high_ami_detail["incremental_ami_revenue_current_policy"]
)

high_ami_rows = []

for tariff_group, group in high_ami_detail.groupby(
    "original_tariff_group",
    observed=True,
):

    group_pv = group["shs_label_numeric"].eq(1)
    group_unregistered = (
        group_pv
        & group["has_registration"].eq(0)
    )

    observed_loss = (
        group["baseline_no_shs_revenue_preferred"].sum()
        - group["observed_adoption_revenue_preferred"].sum()
    )

    current_policy_ami = group[
        "incremental_ami_revenue_current_policy"
    ].sum()

    high_fee_ami = group[
        "incremental_ami_revenue_high_fee"
    ].sum()

    extra_from_fee = group[
        "additional_revenue_from_retaining_high_fee"
    ].sum()

    high_ami_rows.append(
        {
            "tariff_group": tariff_group,
            "pv_households": group.loc[
                group_pv,
                "contract_account_hashed",
            ].nunique(),
            "unregistered_pv_households": group.loc[
                group_unregistered,
                "contract_account_hashed",
            ].nunique(),
            "unregistered_pv_household_months": int(
                group_unregistered.sum()
            ),
            "observed_pv_revenue_loss": observed_loss,
            "incremental_ami_revenue_current_policy": (
                current_policy_ami
            ),
            "incremental_ami_revenue_high_fee": (
                high_fee_ami
            ),
            "additional_revenue_from_retaining_high_fee": (
                extra_from_fee
            ),
            "pct_pv_revenue_loss_offset_high_fee": (
                100 * high_fee_ami / observed_loss
                if observed_loss != 0
                else np.nan
            ),
            "pct_total_residential_baseline_revenue_high_fee": (
                100
                * high_fee_ami
                / total_residential_baseline_revenue
            ),
        }
    )

high_ami_summary = pd.DataFrame(
    high_ami_rows
)

high_total_observed_loss = high_ami_summary[
    "observed_pv_revenue_loss"
].sum()

high_total_current_policy_ami = high_ami_summary[
    "incremental_ami_revenue_current_policy"
].sum()

high_total_high_fee_ami = high_ami_summary[
    "incremental_ami_revenue_high_fee"
].sum()

high_total_extra_fee = high_ami_summary[
    "additional_revenue_from_retaining_high_fee"
].sum()

high_total = pd.DataFrame(
    [
        {
            "tariff_group": "TOTAL",
            "pv_households": high_ami_detail.loc[
                high_ami_detail["shs_label_numeric"].eq(1),
                "contract_account_hashed",
            ].nunique(),
            "unregistered_pv_households": high_ami_detail.loc[
                high_ami_detail["shs_label_numeric"].eq(1)
                & high_ami_detail["has_registration"].eq(0),
                "contract_account_hashed",
            ].nunique(),
            "unregistered_pv_household_months": int(
                (
                    high_ami_detail["shs_label_numeric"].eq(1)
                    & high_ami_detail["has_registration"].eq(0)
                ).sum()
            ),
            "observed_pv_revenue_loss": high_total_observed_loss,
            "incremental_ami_revenue_current_policy": (
                high_total_current_policy_ami
            ),
            "incremental_ami_revenue_high_fee": (
                high_total_high_fee_ami
            ),
            "additional_revenue_from_retaining_high_fee": (
                high_total_extra_fee
            ),
            "pct_pv_revenue_loss_offset_high_fee": (
                100
                * high_total_high_fee_ami
                / high_total_observed_loss
                if high_total_observed_loss != 0
                else np.nan
            ),
            "pct_total_residential_baseline_revenue_high_fee": (
                100
                * high_total_high_fee_ami
                / total_residential_baseline_revenue
            ),
        }
    ]
)

high_ami_summary = pd.concat(
    [
        high_ami_summary,
        high_total,
    ],
    ignore_index=True,
)

print(
    high_ami_summary.to_string(
        index=False
    ),
    flush=True,
)

print(
    "\\nInterpretation: this sensitivity keeps the targeted Home User + "
    "Postpaid registration-compliance population unchanged, but assumes the "
    "City retains the R96.20/month residential AMI fee for Jul-Dec 2023 "
    "instead of reducing it to R5.66/month.",
    flush=True,
)


print(
    "\nSYSTEM-LEVEL CONTEXT FOR TARGETED COMPLIANCE:",
    flush=True,
)

print(
    "Total residential baseline revenue (no-PV counterfactual): "
    f"R{total_residential_baseline_revenue:,.2f}",
    flush=True,
)

print(
    "Total residential observed-adoption revenue: "
    f"R{total_residential_observed_revenue:,.2f}",
    flush=True,
)

print(
    "Targeted incremental AMI revenue: "
    f"R{targeted_incremental_ami:,.2f}",
    flush=True,
)

print(
    "Targeted AMI revenue as % of total residential baseline revenue: "
    f"{targeted_total_result['pct_total_residential_baseline_revenue']:.4f}%",
    flush=True,
)

print(
    "Targeted AMI revenue as % of total residential observed revenue: "
    f"{targeted_total_result['pct_total_residential_observed_revenue']:.4f}%",
    flush=True,
)


###############################################################################
# LIFELINE COMPARISON
###############################################################################

section("LIFELINE PREFERRED VS GROSS RESULTS")

lifeline_compare = pd.concat(
    [
        actual_summary.loc[
            actual_summary[
                "tariff_group"
            ].eq("LifeLine")
        ].assign(
            scenario_short="Actual adoption"
        ),
        universal_summary.loc[
            universal_summary[
                "tariff_group"
            ].eq("LifeLine")
        ].assign(
            scenario_short="Universal adoption"
        ),
    ],
    ignore_index=True,
)

lifeline_columns = [
    "scenario_short",
    "household_months",
    "households",
    "lifeline_rows_exact_fbe",
    "lifeline_rows_gross_fallback",
    "revenue_loss_gross",
    "pct_total_revenue_loss_gross",
    "revenue_loss_preferred",
    "pct_total_revenue_loss_preferred",
]

print(
    lifeline_compare[
        lifeline_columns
    ].to_string(
        index=False
    ),
    flush=True,
)


###############################################################################
# MONTHLY SUMMARY
###############################################################################

section("CREATING MONTHLY SUMMARY")

monthly_rows = []

for scenario_name, detail in [
    (
        "Actual observed SHS adoption",
        actual_detail,
    ),
    (
        "Universal SHS adoption",
        universal_detail,
    ),
]:
    monthly = (
        detail.groupby(
            [
                "month_year",
                "tariff_group",
            ],
            observed=True,
        )
        .agg(
            household_months=(
                "contract_account_hashed",
                "size",
            ),
            baseline_kwh=(
                "baseline_no_shs_kwh",
                "sum",
            ),
            kwh_reduction=(
                "kwh_reduction",
                "sum",
            ),
            revenue_loss_gross=(
                "revenue_loss_gross",
                "sum",
            ),
            revenue_loss_preferred=(
                "revenue_loss_preferred",
                "sum",
            ),
        )
        .reset_index()
    )

    monthly["scenario"] = scenario_name
    monthly_rows.append(monthly)

monthly_summary = pd.concat(
    monthly_rows,
    ignore_index=True,
)


###############################################################################
# SAVE OUTPUTS
###############################################################################

section("SAVING OUTPUTS")

actual_file = (
    OUTPUT_DIR
    / "revenue_loss_actual_adoption_2023_preferred.csv"
)

universal_file = (
    OUTPUT_DIR
    / "revenue_loss_universal_adoption_2023_preferred.csv"
)

lifeline_file = (
    OUTPUT_DIR
    / "lifeline_revenue_preferred_vs_gross_2023.csv"
)

monthly_file = (
    OUTPUT_DIR
    / "revenue_loss_monthly_2023_preferred.csv"
)

qa_file = (
    OUTPUT_DIR
    / "revenue_analysis_2023_household_month_preferred_QA.parquet"
)


tariff_compliance_file = (
    OUTPUT_DIR
    / "revenue_recovered_home_user_tariff_compliance_2023.csv"
)

tariff_compliance_qa_file = (
    OUTPUT_DIR
    / "revenue_home_user_tariff_compliance_2023_QA.parquet"
)


full_registration_file = (
    OUTPUT_DIR
    / "revenue_recovered_full_sseg_registration_ami_compliance_2023.csv"
)

full_registration_qa_file = (
    OUTPUT_DIR
    / "revenue_full_sseg_registration_ami_compliance_2023_QA.parquet"
)


targeted_compliance_file = (
    OUTPUT_DIR
    / "revenue_targeted_homeuser_postpaid_registration_ami_2023.csv"
)


high_ami_sensitivity_file = (
    OUTPUT_DIR
    / "revenue_targeted_homeuser_postpaid_high_ami_fee_2023.csv"
)


actual_summary.to_csv(
    actual_file,
    index=False,
)

universal_summary.to_csv(
    universal_file,
    index=False,
)

lifeline_compare[
    lifeline_columns
].to_csv(
    lifeline_file,
    index=False,
)

monthly_summary.to_csv(
    monthly_file,
    index=False,
)


tariff_compliance_summary.to_csv(
    tariff_compliance_file,
    index=False,
)

sseg_qa_columns = [
    "contract_account_hashed",
    "month_year",
    "original_tariff_group",
    "compliance_tariff_group",
    "shs_label_numeric",
    "compliance_kwh",
    "switched_to_home_user",
    "observed_adoption_revenue_preferred",
    "compliance_total_revenue_preferred",
    "revenue_recovered_preferred",
    "remaining_revenue_loss_preferred",
]

tariff_compliance_detail[
    sseg_qa_columns
].to_parquet(
    tariff_compliance_qa_file,
    index=False,
)


full_registration_summary.to_csv(
    full_registration_file,
    index=False,
)


targeted_compliance_summary.to_csv(
    targeted_compliance_file,
    index=False,
)


high_ami_summary.to_csv(
    high_ami_sensitivity_file,
    index=False,
)

full_registration_qa_columns = [
    "contract_account_hashed",
    "month_year",
    "original_tariff_group",
    "shs_label_numeric",
    "has_registration",
    "ami_fee_rand",
    "observed_registered_ami_revenue",
    "full_compliance_ami_revenue",
    "incremental_ami_revenue",
    "tariff_revenue_recovered",
    "combined_revenue_recovered",
    "remaining_revenue_loss_after_full_compliance",
]

full_registration_detail[
    full_registration_qa_columns
].to_parquet(
    full_registration_qa_file,
    index=False,
)

qa_columns = [
    "contract_account_hashed",
    "month_year",
    "tariff_group",
    "kwh",
    "shs_label_numeric",
    "beta",
    "shs_factor",
    "prior_12m_avg_kwh",
    "fbe_kwh",
    "fbe_history_complete",
    "financial_year",
    "baseline_no_shs_kwh",
    "actual_adoption_kwh",
    "universal_shs_kwh",
]

analysis[
    qa_columns
].to_parquet(
    qa_file,
    index=False,
)

print(
    "Saved actual-adoption results:",
    actual_file,
    flush=True,
)

print(
    "Saved universal-adoption results:",
    universal_file,
    flush=True,
)

print(
    "Saved Lifeline comparison:",
    lifeline_file,
    flush=True,
)

print(
    "Saved monthly results:",
    monthly_file,
    flush=True,
)

print(
    "Saved household-month QA:",
    qa_file,
    flush=True,
)


print(
    "Saved Home User tariff-compliance summary:",
    tariff_compliance_file,
    flush=True,
)

print(
    "Saved Home User tariff-compliance QA:",
    tariff_compliance_qa_file,
    flush=True,
)


print(
    "Saved full SSEG registration/AMI compliance summary:",
    full_registration_file,
    flush=True,
)


print(
    "Saved targeted Home User/Postpaid registration/AMI summary:",
    targeted_compliance_file,
    flush=True,
)


print(
    "Saved targeted high-AMI-fee sensitivity summary:",
    high_ami_sensitivity_file,
    flush=True,
)

print(
    "Saved full SSEG registration/AMI compliance QA:",
    full_registration_qa_file,
    flush=True,
)


###############################################################################
# FINAL REPORT
###############################################################################

section("FINAL 2023 REVENUE RESULTS")

report_columns = [
    "tariff_group",
    "households",
    "kwh_reduction",
    "revenue_loss_gross",
    "pct_total_revenue_loss_gross",
    "revenue_loss_preferred",
    "pct_total_revenue_loss_preferred",
]

print("\nACTUAL ADOPTION:", flush=True)

print(
    actual_summary[
        report_columns
    ].to_string(
        index=False
    ),
    flush=True,
)

print("\nUNIVERSAL ADOPTION:", flush=True)

print(
    universal_summary[
        report_columns
    ].to_string(
        index=False
    ),
    flush=True,
)


print("\nHOME USER TARIFF-CLASS COMPLIANCE:", flush=True)

sseg_report_columns = [
    "original_tariff_group",
    "pv_households",
    "switched_households",
    "observed_revenue_loss_preferred",
    "revenue_recovered_preferred",
    "remaining_revenue_loss_preferred",
    "pct_original_loss_recovered_preferred",
]

print(
    tariff_compliance_summary[
        sseg_report_columns
    ].to_string(
        index=False
    ),
    flush=True,
)


print("\nFULL SSEG REGISTRATION / AMI COMPLIANCE:", flush=True)

full_registration_report_columns = [
    "original_tariff_group",
    "pv_households",
    "registered_pv_households",
    "unregistered_pv_households",
    "tariff_revenue_recovered",
    "incremental_ami_revenue",
    "combined_revenue_recovered",
    "remaining_revenue_loss_after_full_compliance",
    "pct_original_loss_recovered_combined",
]

print(
    full_registration_summary[
        full_registration_report_columns
    ].to_string(
        index=False
    ),
    flush=True,
)


print("\nTARGETED HOME USER + POSTPAID REGISTRATION / AMI COMPLIANCE:", flush=True)

print(
    targeted_compliance_summary.to_string(
        index=False
    ),
    flush=True,
)


print("\nTARGETED REGISTRATION WITH HIGH AMI FEE RETAINED ALL YEAR:", flush=True)

print(
    high_ami_summary.to_string(
        index=False
    ),
    flush=True,
)

print("\nInterpretation notes:", flush=True)

print(
    "1. Preferred results use exact FBE billing for Lifeline rows with "
    "complete prior 12-month history.",
    flush=True,
)

print(
    "2. Lifeline rows without enough history remain in the full sample "
    "and use gross-of-FBE billing as a conservative fallback.",
    flush=True,
)

print(
    "3. Gross results ignore FBE everywhere and are retained as a "
    "sensitivity check.",
    flush=True,
)

print(
    "4. Home User fixed charges are included in total-revenue denominators "
    "and cancel from the SHS/no-SHS difference.",
    flush=True,
)

print(
    "5. Feed-in tariff payments are not included.",
    flush=True,
)

print(
    "6. Revenue values are in South African rand (ZAR).",
    flush=True,
)


print(
    "7. The Home User tariff-compliance counterfactual keeps observed PV consumption "
    "unchanged and switches detected LifeLine and Domestic PV "
    "household-months to the Home User tariff only while PV is present.",
    flush=True,
)

print(
    "8. Home User and Postpaid Home User PV observations are unchanged "
    "in the tariff-compliance counterfactual because the current revenue script "
    "already applies the same Home User energy rates and fixed charge to "
    "those groups.",
    flush=True,
)

print(
    "9. The full-registration counterfactual adds the applicable residential AMI fee "
    "to detected PV household-months that are not already registered in the official data.",
    flush=True,
)

print(
    "10. Existing registered PV household-months are reported with their implied AMI revenue, "
    "but this is not counted as newly recovered revenue under full registration.",
    flush=True,
)

print(
    "11. Feed-in tariff payments remain excluded because export quantities are not observed.",
    flush=True,
)

print(
    "12. AMI fees used are VAT-inclusive: R96.20/month for Jan-Jun 2023 "
    "and R5.66/month for Jul-Dec 2023.",
    flush=True,
)


print(
    "13. The targeted Home User + Postpaid scenario leaves Lifeline and "
    "Domestic unchanged and counts only incremental AMI revenue from "
    "currently unregistered detected PV household-months in those two "
    "higher-revenue tariff groups.",
    flush=True,
)


print(
    "14. The targeted compliance output also reports incremental AMI revenue "
    "as a percentage of total residential electricity revenue, allowing the "
    "policy effect to be interpreted at the municipal residential-system level "
    "rather than only relative to PV-associated revenue loss.",
    flush=True,
)


print(
    "15. The high-AMI-fee sensitivity keeps the targeted Home User + Postpaid "
    "registration population unchanged but applies R96.20/month throughout "
    "2023; it reports both total AMI revenue under that policy and the "
    "additional revenue attributable specifically to retaining the higher fee "
    "in Jul-Dec.",
    flush=True,
)

print("\nProcessing complete!", flush=True)
