import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import POSTPAID_PANEL, PREPAID_PANEL, REGRESSION_DIR

import pandas as pd
import numpy as np
from dataclasses import dataclass
from linearmodels.panel.data import PanelData
from scipy import stats


###############################################################################
# CONFIG
###############################################################################

PREPAID_FILE = (
    str(PREPAID_PANEL)
)

POSTPAID_FILE = (
    str(POSTPAID_PANEL)
)

OUTPUT_DIRECTORY = Path(
    REGRESSION_DIR
)

END_DATE = pd.Timestamp("2023-05-31")

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

COLS_NEEDED = [
    "contract_account_hashed",
    "month_year",
    "shs_label_numeric",
    "kwh",
    "Area_number",
    "tariff_category",
    "PV_kW",
    "max_contracts_building_panel",
    "total_duration_hr",
    "total_peak_hr",
    "total_standard_hr",
    "total_off_peak_hr",
]


###############################################################################
# LOAD DATA
###############################################################################

print("\n" + "=" * 90)
print("LOADING PREPAID AND POSTPAID DATA")
print("=" * 90)

prepaid = pd.read_parquet(PREPAID_FILE, columns=COLS_NEEDED)
postpaid = pd.read_parquet(POSTPAID_FILE, columns=COLS_NEEDED)

print("Prepaid rows:", f"{len(prepaid):,}")
print("Postpaid rows:", f"{len(postpaid):,}")

combined = pd.concat([prepaid, postpaid], ignore_index=True)

print("Combined rows:", f"{len(combined):,}")
print(
    "Combined unique households:",
    f"{combined['contract_account_hashed'].nunique():,}",
)
print("Exact duplicate rows:", f"{combined.duplicated().sum():,}")

combined = combined.drop_duplicates().copy()

pv_rows_above_cutoff = (
    combined["shs_label_numeric"].eq(1)
    & combined["max_contracts_building_panel"].gt(5)
).sum()

print(
    "PV rows above 5-contract building cutoff:",
    f"{pv_rows_above_cutoff:,}",
)

if pv_rows_above_cutoff > 0:
    raise ValueError(
        "PV observations remain in buildings with more than 5 contracts."
    )

###############################################################################
# BASIC SAMPLE RESTRICTIONS
###############################################################################

print("\n" + "=" * 90)
print("APPLYING SAMPLE RESTRICTIONS")
print("=" * 90)

combined["month_year"] = pd.to_datetime(
    combined["month_year"],
    errors="coerce",
)

numeric_columns = [
    "kwh",
    "Area_number",
    "shs_label_numeric",
    "PV_kW",
    "total_duration_hr",
    "total_peak_hr",
    "total_standard_hr",
    "total_off_peak_hr",
]

for col in numeric_columns:
    combined[col] = pd.to_numeric(
        combined[col],
        errors="coerce",
    )

combined = combined.loc[
    combined["month_year"].notna()
    & combined["month_year"].le(END_DATE)
    & combined["kwh"].notna()
    & combined["kwh"].ge(0)
    & combined["Area_number"].between(1, 16)
].copy()

print("Observations after restrictions:", f"{len(combined):,}")
print(
    "Unique households after restrictions:",
    f"{combined['contract_account_hashed'].nunique():,}",
)


###############################################################################
# OUTAGE QA AND STAGE-1 EQUIVALENTS
###############################################################################

print("\n" + "=" * 90)
print("CHECKING OUTAGE EXPOSURE")
print("=" * 90)

OUTAGE_HOUR_COLS = [
    "total_duration_hr",
    "total_peak_hr",
    "total_standard_hr",
    "total_off_peak_hr",
]

print("\nMissing outage exposure after restricting to blocks 1-16:")
print(combined[OUTAGE_HOUR_COLS].isna().sum())

if combined[OUTAGE_HOUR_COLS].isna().any().any():
    raise ValueError(
        "Missing load-shedding exposure remains after restricting to blocks 1-16."
    )

tou_difference = (
    combined["total_peak_hr"]
    + combined["total_standard_hr"]
    + combined["total_off_peak_hr"]
    - combined["total_duration_hr"]
)

max_tou_difference = tou_difference.abs().max()

print(
    "\nMaximum Peak + Standard + Off-peak minus Total discrepancy:",
    f"{max_tou_difference:.10f}",
)

if max_tou_difference > 1e-8:
    raise ValueError(
        "Peak, standard, and off-peak outage durations do not sum to total duration."
    )

combined["total_duration_stage1"] = combined["total_duration_hr"] / 2.5
combined["total_peak_stage1"] = combined["total_peak_hr"] / 2.5
combined["total_standard_stage1"] = combined["total_standard_hr"] / 2.5
combined["total_offpeak_stage1"] = combined["total_off_peak_hr"] / 2.5


###############################################################################
# BINARY PV QA
###############################################################################

print("\n" + "=" * 90)
print("CHECKING BINARY PV INDICATOR")
print("=" * 90)

invalid_shs = ~combined["shs_label_numeric"].isin([0, 1])

print(
    "Rows with invalid/missing shs_label_numeric:",
    f"{invalid_shs.sum():,}",
)

if invalid_shs.any():
    raise ValueError(
        "shs_label_numeric must be non-missing and equal to 0 or 1."
    )

combined["shs_label_numeric"] = combined["shs_label_numeric"].astype(int)

print(
    "PV household-month share:",
    f"{combined['shs_label_numeric'].mean():.6f}",
)

ever_pv = (
    combined
    .groupby("contract_account_hashed")["shs_label_numeric"]
    .max()
)

print(
    "Unique households ever observed with PV:",
    f"{(ever_pv == 1).sum():,}",
)


###############################################################################
# CONSTRUCT BINARY INTERACTIONS
###############################################################################

print("\n" + "=" * 90)
print("CONSTRUCTING BINARY INTERACTIONS")
print("=" * 90)

combined["interaction_duration_stage1_shs"] = (
    combined["total_duration_stage1"]
    * combined["shs_label_numeric"]
)

combined["interaction_peak_stage1_shs"] = (
    combined["total_peak_stage1"]
    * combined["shs_label_numeric"]
)

combined["interaction_standard_stage1_shs"] = (
    combined["total_standard_stage1"]
    * combined["shs_label_numeric"]
)

combined["interaction_offpeak_stage1_shs"] = (
    combined["total_offpeak_stage1"]
    * combined["shs_label_numeric"]
)


###############################################################################
# BUILD BINARY REGRESSION SAMPLE
###############################################################################

print("\n" + "=" * 90)
print("BUILDING BINARY REGRESSION SAMPLE")
print("=" * 90)

BINARY_REQUIRED = [
    "contract_account_hashed",
    "month_year",
    "kwh",
    "Area_number",
    "shs_label_numeric",
    "total_duration_stage1",
    "total_peak_stage1",
    "total_standard_stage1",
    "total_offpeak_stage1",
    "interaction_duration_stage1_shs",
    "interaction_peak_stage1_shs",
    "interaction_standard_stage1_shs",
    "interaction_offpeak_stage1_shs",
]

print("\nMissing values in binary regression variables:")
print(
    combined[BINARY_REQUIRED]
    .isna()
    .sum()
)

binary = combined.dropna(
    subset=BINARY_REQUIRED
).copy()

binary = (
    binary
    .sort_values(
        ["contract_account_hashed", "month_year"]
    )
    .reset_index(drop=True)
)

binary_duplicate_mask = binary.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ],
    keep=False,
)

print(
    "\nBinary sample rows involved in duplicate account-months:",
    f"{binary_duplicate_mask.sum():,}",
)

if binary_duplicate_mask.any():

    duplicate_rows = binary.loc[
        binary_duplicate_mask
    ].copy()

    binary_relevant_cols = [
        "kwh",
        "Area_number",
        "shs_label_numeric",
        "total_duration_stage1",
        "total_peak_stage1",
        "total_standard_stage1",
        "total_offpeak_stage1",
    ]

    duplicate_variation = (
        duplicate_rows
        .groupby(
            [
                "contract_account_hashed",
                "month_year",
            ]
        )[binary_relevant_cols]
        .nunique(dropna=False)
    )

    differences_by_variable = (
        duplicate_variation
        .gt(1)
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "\nDuplicated binary account-months differing "
        "on regression variables:"
    )

    print(
        differences_by_variable[
            differences_by_variable > 0
        ]
    )

    if differences_by_variable.gt(0).any():
        raise ValueError(
            "Some duplicate binary account-months differ "
            "on regression variables."
        )

    binary = binary.drop_duplicates(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep="first",
    ).copy()

remaining_binary_duplicates = binary.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ]
).sum()

print(
    "Remaining duplicate binary account-months:",
    f"{remaining_binary_duplicates:,}",
)

if remaining_binary_duplicates > 0:
    raise ValueError(
        "Duplicate account-months remain in binary sample."
    )

print("\n" + "=" * 90)
print("BINARY SAMPLE QA")
print("=" * 90)

print(
    "Binary observations:",
    f"{len(binary):,}",
)

print(
    "Binary unique households:",
    f"{binary['contract_account_hashed'].nunique():,}",
)

print(
    "Binary time periods:",
    f"{binary['month_year'].nunique():,}",
)

print(
    "Binary blocks:",
    sorted(
        binary["Area_number"]
        .astype(int)
        .unique()
        .tolist()
    ),
)

binary = (
    binary
    .set_index(
        [
            "contract_account_hashed",
            "month_year",
        ]
    )
    .sort_index()
)

binary_y_log = np.log(
    binary[["kwh"]].astype(float)
    + 0.01
)

binary_block_clusters = (
    binary[["Area_number"]]
    .astype(int)
)


###############################################################################
# OPTIMIZED TWO-WAY FE HELPERS
###############################################################################

@dataclass
class CachedFEResult:
    model_name: str
    params: pd.Series
    std_errors: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    conf_low: pd.Series
    conf_high: pd.Series
    nobs: int
    n_entities: int
    n_periods: int
    df_resid: int
    n_clusters: int

    @property
    def summary(self) -> str:
        table = pd.DataFrame(
            {
                "Parameter": self.params,
                "Std. Err.": self.std_errors,
                "T-stat": self.tstats,
                "P-value": self.pvalues,
                "Lower CI": self.conf_low,
                "Upper CI": self.conf_high,
            }
        )

        header = [
            self.model_name,
            f"Observations: {self.nobs:,}",
            f"Entities: {self.n_entities:,}",
            f"Time periods: {self.n_periods:,}",
            f"Cluster groups: {self.n_clusters:,}",
            f"Residual df: {self.df_resid:,}",
            "Included effects: Entity, Time",
            "",
        ]

        return "\n".join(header) + table.to_string(
            float_format=lambda x: f"{x:.8g}"
        )


def prepare_two_way_within(
    data,
    dependent,
    variables,
    family_name,
):
    """
    Two-way demean the dependent variable and all regressors ONCE.

    PanelData.demean('both', low_memory=True) uses the same low-memory
    two-way fixed-effect transformation used by PanelOLS for an
    unbalanced panel.  Reusing the transformed variables avoids
    repeating this expensive operation for every model specification.
    """

    print("\n" + "=" * 90)
    print(f"PRECOMPUTING TWO-WAY FIXED EFFECTS: {family_name}")
    print("=" * 90)

    dep_name = "__dependent__"

    frame = pd.concat(
        [
            dependent.rename(columns={dependent.columns[0]: dep_name}),
            data[variables],
        ],
        axis=1,
    ).astype(float)

    transformed = (
        PanelData(frame)
        .demean(
            "both",
            low_memory=True,
        )
        .dataframe
    )

    n_entities = int(
        data.index.get_level_values(
            "contract_account_hashed"
        ).nunique()
    )

    n_periods = int(
        data.index.get_level_values(
            "month_year"
        ).nunique()
    )

    print(
        "Within transformation complete:",
        f"{len(transformed):,} rows,",
        f"{len(variables)} regressors prepared.",
        flush=True,
    )

    return transformed, n_entities, n_periods


def run_model_from_within(
    transformed,
    clusters,
    model_name,
    control_vars,
    n_entities,
    n_periods,
):
    """
    Fit OLS to already two-way-demeaned data.

    Coefficients are the same within-estimator coefficients as PanelOLS
    with entity_effects=True and time_effects=True.

    The one-way clustered covariance is computed using the same
    absorbed-effect degrees-of-freedom scaling used by PanelOLS:
        scale = N / (N - absorbed_effects - K)
    where absorbed_effects = N_entities + N_periods - 1.
    """

    print("\n" + "=" * 90)
    print(model_name)
    print("=" * 90)

    y = transformed["__dependent__"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    X = transformed[control_vars].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    nobs, nvar = X.shape

    absorbed_effects = (
        n_entities
        + n_periods
        - 1
    )

    df_resid = (
        nobs
        - absorbed_effects
        - nvar
    )

    if df_resid <= 0:
        raise ValueError(
            f"Non-positive residual degrees of freedom in {model_name}."
        )

    xtx = X.T @ X
    xty = X.T @ y

    rank = np.linalg.matrix_rank(xtx)

    if rank < nvar:
        raise ValueError(
            f"{model_name}: transformed regressor matrix is rank deficient "
            f"(rank {rank}, columns {nvar})."
        )

    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ xty

    residuals = y - X @ beta

    cluster_values = (
        clusters.iloc[:, 0]
        .reindex(transformed.index)
        .to_numpy()
    )

    if pd.isna(cluster_values).any():
        raise ValueError(
            f"{model_name}: missing cluster values after alignment."
        )

    cluster_codes, cluster_levels = pd.factorize(
        cluster_values,
        sort=True,
    )

    n_clusters = len(cluster_levels)

    if n_clusters < 2:
        raise ValueError(
            f"{model_name}: fewer than two cluster groups."
        )

    # Cluster score for each block:
    #     S_g = sum_i X_i * residual_i
    # With only 16 blocks, bincount avoids constructing a giant
    # N x K score DataFrame.
    cluster_scores = np.empty(
        (n_clusters, nvar),
        dtype=np.float64,
    )

    for j in range(nvar):
        cluster_scores[:, j] = np.bincount(
            cluster_codes,
            weights=X[:, j] * residuals,
            minlength=n_clusters,
        )

    meat = (
        cluster_scores.T
        @ cluster_scores
    )

    # Matches PanelOLS's default debiased clustered covariance scaling
    # when entity and time effects are absorbed.
    df_scale = (
        nobs
        / df_resid
    )

    covariance = (
        df_scale
        * xtx_inv
        @ meat
        @ xtx_inv
    )

    std_errors = np.sqrt(
        np.diag(covariance)
    )

    tstats = (
        beta
        / std_errors
    )

    pvalues = (
        2
        * stats.t.sf(
            np.abs(tstats),
            df=df_resid,
        )
    )

    critical_value = stats.t.ppf(
        0.975,
        df=df_resid,
    )

    conf_low = (
        beta
        - critical_value * std_errors
    )

    conf_high = (
        beta
        + critical_value * std_errors
    )

    result = CachedFEResult(
        model_name=model_name,
        params=pd.Series(
            beta,
            index=control_vars,
        ),
        std_errors=pd.Series(
            std_errors,
            index=control_vars,
        ),
        tstats=pd.Series(
            tstats,
            index=control_vars,
        ),
        pvalues=pd.Series(
            pvalues,
            index=control_vars,
        ),
        conf_low=pd.Series(
            conf_low,
            index=control_vars,
        ),
        conf_high=pd.Series(
            conf_high,
            index=control_vars,
        ),
        nobs=nobs,
        n_entities=n_entities,
        n_periods=n_periods,
        df_resid=df_resid,
        n_clusters=n_clusters,
    )

    print(result.summary, flush=True)

    return result


###############################################################################
# MODELS 1-4: FULL BINARY SAMPLE
###############################################################################

BINARY_WITHIN_VARS = [
    "total_duration_stage1",
    "total_peak_stage1",
    "total_standard_stage1",
    "total_offpeak_stage1",
    "shs_label_numeric",
    "interaction_duration_stage1_shs",
    "interaction_peak_stage1_shs",
    "interaction_standard_stage1_shs",
    "interaction_offpeak_stage1_shs",
]

(
    binary_within,
    binary_n_entities,
    binary_n_periods,
) = prepare_two_way_within(
    binary,
    binary_y_log,
    BINARY_WITHIN_VARS,
    "BINARY MODELS 1-4",
)

combine_outage1 = run_model_from_within(
    binary_within,
    binary_block_clusters,
    "MODEL 1: TOTAL OUTAGE ONLY",
    [
        "total_duration_stage1",
    ],
    binary_n_entities,
    binary_n_periods,
)

combine_outage2 = run_model_from_within(
    binary_within,
    binary_block_clusters,
    "MODEL 2: OUTAGE BY TIME OF USE",
    [
        "total_peak_stage1",
        "total_standard_stage1",
        "total_offpeak_stage1",
    ],
    binary_n_entities,
    binary_n_periods,
)

outage_shs1 = run_model_from_within(
    binary_within,
    binary_block_clusters,
    "MODEL 3: BINARY PV + TOTAL OUTAGE",
    [
        "total_duration_stage1",
        "shs_label_numeric",
        "interaction_duration_stage1_shs",
    ],
    binary_n_entities,
    binary_n_periods,
)

outage_shs2 = run_model_from_within(
    binary_within,
    binary_block_clusters,
    "MODEL 4: BINARY PV + OUTAGE BY TIME OF USE",
    [
        "total_peak_stage1",
        "total_standard_stage1",
        "total_offpeak_stage1",
        "shs_label_numeric",
        "interaction_peak_stage1_shs",
        "interaction_standard_stage1_shs",
        "interaction_offpeak_stage1_shs",
    ],
    binary_n_entities,
    binary_n_periods,
)


###############################################################################
# BUILD CONTINUOUS-CAPACITY SAMPLE
###############################################################################

print("\n" + "=" * 90)
print("BUILDING CONTINUOUS-CAPACITY SAMPLE")
print("=" * 90)


continuous = combined.copy()

continuous["PV_kW"] = pd.to_numeric(
    continuous["PV_kW"],
    errors="coerce",
)

negative_capacity = (
    continuous["PV_kW"].notna()
    & continuous["PV_kW"].lt(0)
)

print(
    "Rows with negative PV capacity:",
    f"{negative_capacity.sum():,}",
)

if negative_capacity.any():
    raise ValueError(
        "Negative PV capacity values found."
    )

# Step 8 should provide capacity for every observation classified as PV.
pv_missing_capacity = (
    continuous["shs_label_numeric"].eq(1)
    & continuous["PV_kW"].isna()
)

print(
    "PV household-months with missing Step-8 PV_kW:",
    f"{pv_missing_capacity.sum():,}",
)

if pv_missing_capacity.any():
    raise ValueError(
        "Some PV household-months have missing Step-8 PV_kW."
    )

# Genuine non-PV household-months have zero installed PV capacity.
non_pv = continuous["shs_label_numeric"].eq(0)

continuous.loc[
    non_pv & continuous["PV_kW"].isna(),
    "PV_kW",
] = 0.0

remaining_missing_capacity = (
    continuous["PV_kW"].isna()
)

print(
    "Remaining rows with missing PV_kW:",
    f"{remaining_missing_capacity.sum():,}",
)

if remaining_missing_capacity.any():
    raise ValueError(
        "PV_kW remains missing after assigning zero capacity "
        "to non-PV household-months."
    )

non_pv_positive_capacity = (
    continuous["shs_label_numeric"].eq(0)
    & continuous["PV_kW"].gt(0)
)

print(
    "Non-PV rows with positive capacity:",
    f"{non_pv_positive_capacity.sum():,}",
)

if non_pv_positive_capacity.any():
    raise ValueError(
        "Some non-PV observations have positive PV capacity."
    )

positive_capacity = continuous.loc[
    continuous["PV_kW"] > 0,
    "PV_kW",
]

if positive_capacity.empty:
    raise ValueError(
        "No positive PV capacity observations remain."
    )

print("\nPositive-capacity quantiles:")
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
    .quantile(0.99)
)

print(
    f"\n99th percentile of positive PV capacity: "
    f"{pv_upper_bound:.6f} kW"
)

continuous["PV_kW_trimmed"] = (
    continuous["PV_kW"]
    .clip(
        lower=0,
        upper=pv_upper_bound,
    )
)

continuous["interaction_duration_stage1_PV"] = (
    continuous["total_duration_stage1"]
    * continuous["PV_kW_trimmed"]
)

continuous["interaction_peak_stage1_PV"] = (
    continuous["total_peak_stage1"]
    * continuous["PV_kW_trimmed"]
)

continuous["interaction_standard_stage1_PV"] = (
    continuous["total_standard_stage1"]
    * continuous["PV_kW_trimmed"]
)

continuous["interaction_offpeak_stage1_PV"] = (
    continuous["total_offpeak_stage1"]
    * continuous["PV_kW_trimmed"]
)

CONTINUOUS_REQUIRED = [
    "contract_account_hashed",
    "month_year",
    "kwh",
    "Area_number",
    "PV_kW_trimmed",
    "total_duration_stage1",
    "total_peak_stage1",
    "total_standard_stage1",
    "total_offpeak_stage1",
    "interaction_duration_stage1_PV",
    "interaction_peak_stage1_PV",
    "interaction_standard_stage1_PV",
    "interaction_offpeak_stage1_PV",
]

print(
    "\nMissing values before continuous-sample drop:"
)

print(
    continuous[
        CONTINUOUS_REQUIRED
    ]
    .isna()
    .sum()
)

continuous = (
    continuous
    .dropna(
        subset=CONTINUOUS_REQUIRED
    )
    .copy()
)

continuous = (
    continuous
    .sort_values(
        ["contract_account_hashed", "month_year"]
    )
    .reset_index(drop=True)
)

continuous_duplicate_mask = continuous.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ],
    keep=False,
)

print(
    "\nContinuous sample rows involved in duplicate account-months:",
    f"{continuous_duplicate_mask.sum():,}",
)

if continuous_duplicate_mask.any():

    duplicate_rows = continuous.loc[
        continuous_duplicate_mask
    ].copy()

    continuous_relevant_cols = [
        "kwh",
        "Area_number",
        "PV_kW_trimmed",
        "total_duration_stage1",
        "total_peak_stage1",
        "total_standard_stage1",
        "total_offpeak_stage1",
    ]

    duplicate_variation = (
        duplicate_rows
        .groupby(
            [
                "contract_account_hashed",
                "month_year",
            ]
        )[continuous_relevant_cols]
        .nunique(dropna=False)
    )

    differences_by_variable = (
        duplicate_variation
        .gt(1)
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "\nDuplicated continuous account-months differing "
        "on regression variables:"
    )

    print(
        differences_by_variable[
            differences_by_variable > 0
        ]
    )

    if differences_by_variable.gt(0).any():
        raise ValueError(
            "Some duplicate continuous account-months differ "
            "on regression variables."
        )

    continuous = continuous.drop_duplicates(
        subset=[
            "contract_account_hashed",
            "month_year",
        ],
        keep="first",
    ).copy()

remaining_continuous_duplicates = continuous.duplicated(
    subset=[
        "contract_account_hashed",
        "month_year",
    ]
).sum()

print(
    "Remaining duplicate continuous account-months:",
    f"{remaining_continuous_duplicates:,}",
)

if remaining_continuous_duplicates > 0:
    raise ValueError(
        "Duplicate account-months remain in continuous sample."
    )

print("\n" + "=" * 90)
print("CONTINUOUS SAMPLE QA")
print("=" * 90)

print(
    "Continuous observations:",
    f"{len(continuous):,}",
)

print(
    "Continuous unique households:",
    f"{continuous['contract_account_hashed'].nunique():,}",
)

print(
    "Continuous time periods:",
    f"{continuous['month_year'].nunique():,}",
)

print(
    "Continuous blocks:",
    sorted(
        continuous["Area_number"]
        .astype(int)
        .unique()
        .tolist()
    ),
)

continuous = (
    continuous
    .set_index(
        [
            "contract_account_hashed",
            "month_year",
        ]
    )
    .sort_index()
)

continuous_y_log = np.log(
    continuous[["kwh"]].astype(float)
    + 0.01
)

continuous_block_clusters = (
    continuous[["Area_number"]]
    .astype(int)
)

CONTINUOUS_WITHIN_VARS = [
    "total_duration_stage1",
    "total_peak_stage1",
    "total_standard_stage1",
    "total_offpeak_stage1",
    "PV_kW_trimmed",
    "interaction_duration_stage1_PV",
    "interaction_peak_stage1_PV",
    "interaction_standard_stage1_PV",
    "interaction_offpeak_stage1_PV",
]

(
    continuous_within,
    continuous_n_entities,
    continuous_n_periods,
) = prepare_two_way_within(
    continuous,
    continuous_y_log,
    CONTINUOUS_WITHIN_VARS,
    "CONTINUOUS-CAPACITY MODELS 5-6",
)

outage_PV_kW1 = run_model_from_within(
    continuous_within,
    continuous_block_clusters,
    "MODEL 5: CONTINUOUS PV CAPACITY + TOTAL OUTAGE",
    [
        "total_duration_stage1",
        "PV_kW_trimmed",
        "interaction_duration_stage1_PV",
    ],
    continuous_n_entities,
    continuous_n_periods,
)

outage_PV_kW2 = run_model_from_within(
    continuous_within,
    continuous_block_clusters,
    "MODEL 6: CONTINUOUS PV CAPACITY + OUTAGE BY TIME OF USE",
    [
        "total_peak_stage1",
        "total_standard_stage1",
        "total_offpeak_stage1",
        "PV_kW_trimmed",
        "interaction_peak_stage1_PV",
        "interaction_standard_stage1_PV",
        "interaction_offpeak_stage1_PV",
    ],
    continuous_n_entities,
    continuous_n_periods,
)


###############################################################################
# SAVE MODEL SUMMARIES
###############################################################################

print("\n" + "=" * 90)
print("SAVING MODEL SUMMARIES")
print("=" * 90)

summary_txt = (
    OUTPUT_DIRECTORY
    / "allhouseholds_blockclustered_models_09132026.txt"
)

model_results = [
    ("MODEL 1: TOTAL OUTAGE ONLY", combine_outage1),
    ("MODEL 2: OUTAGE BY TIME OF USE", combine_outage2),
    ("MODEL 3: BINARY PV + TOTAL OUTAGE", outage_shs1),
    ("MODEL 4: BINARY PV + OUTAGE BY TIME OF USE", outage_shs2),
    ("MODEL 5: CONTINUOUS PV CAPACITY + TOTAL OUTAGE", outage_PV_kW1),
    ("MODEL 6: CONTINUOUS PV CAPACITY + OUTAGE BY TIME OF USE", outage_PV_kW2),
]

with open(summary_txt, "w") as output_file:
    output_file.write(
        f"BINARY SAMPLE\\nObservations: {len(binary):,}\\n"
        f"Households: {binary.index.get_level_values('contract_account_hashed').nunique():,}\\n\\n"
    )
    output_file.write(
        f"CONTINUOUS SAMPLE\\nObservations: {len(continuous):,}\\n"
        f"Households: {continuous.index.get_level_values('contract_account_hashed').nunique():,}\\n"
        f"Positive-capacity 99th-percentile winsorization cutoff: {pv_upper_bound:.6f} kW\\n\\n"
    )
    for model_name, result in model_results:
        output_file.write(
            "=" * 90
            + "\n"
            + model_name
            + "\n"
            + "=" * 90
            + "\n"
        )
        output_file.write(str(result.summary))
        output_file.write("\n\n")

print("Saved model summaries to:")
print(summary_txt)

print("\nProcessing complete!")
