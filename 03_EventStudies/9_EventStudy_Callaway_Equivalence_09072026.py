# Dose-response event study using the updated August 2026 Cape Town workflow.
#
# Key features:
#   1. Uses ALL never-treated households (no control subsampling).
#   2. Uses corrected/backfilled SHS timing from Step 8.
#   3. Does NOT re-aggregate household-month observations.
#   4. Estimates the outcome regression on never-treated households.
#   5. Re-estimates the outcome regression inside every household bootstrap draw.
#   6. Implements the bootstrap with household multiplicity weights rather than
#      physically duplicating millions of panel rows.
#   7. Uses a sparse/chunked fixed-effect regression to keep memory manageable.
#   8. Preserves pre-treatment event-study estimates relative to g-1; event
#      time -1 is the omitted/reference period.

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, optimize


###############################################################################
# PATHS / PARAMETERS
###############################################################################

PREPAID_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/prepaid_cleaned_vector_noagg.parquet"
)

POSTPAID_FILE = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/postpaid_cleaned_vector_noagg.parquet"
)

PLOT_DIR = (
    "/home/ey53/vscode-server-backup/CapeTown_Workflow/"
    "10_Plots_09182026"
)

END_DATE = pd.Timestamp("2023-12-31")
TREATMENT_CUTOFF = pd.Period("2018-06", freq="M")

BOOTSTRAP_ITERATIONS = 200
RANDOM_SEED = 42

EVENT_MIN = -12
EVENT_MAX = 36

# Chunk size for the sparse first-stage outcome model.
MODEL_CHUNK_SIZE = 500_000

# ------------------------------------------------------------------
# Long-run bootstrap safety / equivalence-test settings
# ------------------------------------------------------------------
# Every completed bootstrap iteration is written immediately to disk.
# If the job stops, rerun the same script and completed iterations will
# automatically be skipped.
BOOTSTRAP_CHECKPOINT_DIR = os.path.join(
    PLOT_DIR,
    "bootstrap_checkpoints_09182026_max5",
)

BOOTSTRAP_DRAWS_FILE = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_bootstrap_09182026_max5.parquet",
)

EQUIVALENCE_OUTPUT_FILE = os.path.join(
    PLOT_DIR,
    "pretrend_equivalence_max_09182026_max5.csv",
)

EQUIVALENCE_ALPHA = 0.05

# Cheap Gaussian simulation performed AFTER the expensive household bootstrap.
# This implements the maximum-norm equivalence test using the general-estimator
# Gaussian approximation discussed by Dette & Schumann (2024), with the
# covariance matrix estimated from the household-cluster bootstrap draws.
EQUIVALENCE_SIM_DRAWS = 50_000

# Also report whether these substantively useful log-point bounds are supported.
EQUIVALENCE_THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
]

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(BOOTSTRAP_CHECKPOINT_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


###############################################################################
# HELPERS
###############################################################################

def section(title: str) -> None:
    print("\n" + "=" * 90, flush=True)
    print(title, flush=True)
    print("=" * 90, flush=True)


def assert_unique_account_month(df: pd.DataFrame, name: str) -> None:
    duplicate_rows = int(
        df.duplicated(
            subset=["contract_account_hashed", "month_year"],
            keep=False,
        ).sum()
    )

    print(
        f"{name} rows involved in duplicate account-months:",
        f"{duplicate_rows:,}",
        flush=True,
    )

    if duplicate_rows > 0:
        raise ValueError(
            f"{name} is not unique by contract_account_hashed and month_year."
        )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


###############################################################################
# SPARSE OUTCOME-REGRESSION DESIGN
###############################################################################

MODEL_CATEGORICAL_COLUMNS = [
    "Area_number_model",
    "tariff_model",
    "month_model",
]


def make_design_spec(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Fix category levels once using the full analysis panel.

    The first level of each categorical variable is the omitted reference
    category. Missing block/tariff values are represented explicitly.
    """
    spec = {}

    for column in MODEL_CATEGORICAL_COLUMNS:
        levels = sorted(
            df[column]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(levels) == 0:
            raise ValueError(
                f"No levels found for {column}."
            )

        spec[column] = levels

    return spec


def design_column_count(
    design_spec: dict[str, list[str]],
) -> int:
    # Intercept + K-1 indicators for each categorical variable.
    return 1 + sum(
        len(levels) - 1
        for levels in design_spec.values()
    )


def make_sparse_design(
    frame: pd.DataFrame,
    design_spec: dict[str, list[str]],
) -> sparse.csr_matrix:
    """
    Construct an intercept + treatment-coded sparse dummy matrix.
    """
    n_rows = len(frame)
    n_columns = design_column_count(
        design_spec
    )

    row_parts = [
        np.arange(
            n_rows,
            dtype=np.int64,
        )
    ]
    col_parts = [
        np.zeros(
            n_rows,
            dtype=np.int64,
        )
    ]
    data_parts = [
        np.ones(
            n_rows,
            dtype=np.float64,
        )
    ]

    column_offset = 1

    for column in MODEL_CATEGORICAL_COLUMNS:
        levels = design_spec[column]

        codes = pd.Categorical(
            frame[column].astype(str),
            categories=levels,
        ).codes

        if (codes < 0).any():
            raise ValueError(
                f"Unknown category encountered in {column}."
            )

        # code 0 is the omitted category.
        use = codes > 0

        if use.any():
            row_parts.append(
                np.flatnonzero(use)
            )
            col_parts.append(
                column_offset
                + codes[use]
                - 1
            )
            data_parts.append(
                np.ones(
                    int(use.sum()),
                    dtype=np.float64,
                )
            )

        column_offset += (
            len(levels) - 1
        )

    row_indices = np.concatenate(
        row_parts
    )
    col_indices = np.concatenate(
        col_parts
    )
    values = np.concatenate(
        data_parts
    )

    return sparse.csr_matrix(
        (
            values,
            (
                row_indices,
                col_indices,
            ),
        ),
        shape=(
            n_rows,
            n_columns,
        ),
        dtype=np.float64,
    )


def fit_outcome_model(
    panel_df: pd.DataFrame,
    design_spec: dict[str, list[str]],
    household_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit the never-treated outcome regression:

        y_it = beta_0 + block FE + tariff FE + month-year FE + error_it

    If household_weights is provided, rows receive the multiplicity of their
    household in the cluster bootstrap draw. This exactly represents a
    household bootstrap without physically duplicating the panel.

    Returns the estimated coefficient vector.
    """
    p = design_column_count(
        design_spec
    )

    xtwx = np.zeros(
        (p, p),
        dtype=np.float64,
    )

    xtwy = np.zeros(
        p,
        dtype=np.float64,
    )

    n_control_rows_used = 0

    for start in range(
        0,
        len(panel_df),
        MODEL_CHUNK_SIZE,
    ):
        chunk = panel_df.iloc[
            start:start + MODEL_CHUNK_SIZE
        ]

        use = (
            chunk["never_treated"].to_numpy()
            & chunk["y"].notna().to_numpy()
        )

        if household_weights is None:
            weights = np.ones(
                len(chunk),
                dtype=np.float64,
            )
        else:
            weights = household_weights[
                chunk["hh_code"].to_numpy(
                    dtype=np.int64
                )
            ].astype(
                np.float64,
                copy=False,
            )

            use &= weights > 0

        if not use.any():
            continue

        model_chunk = chunk.loc[
            use,
            MODEL_CATEGORICAL_COLUMNS,
        ]

        y_chunk = chunk.loc[
            use,
            "y",
        ].to_numpy(
            dtype=np.float64
        )

        w_chunk = weights[use]

        X_chunk = make_sparse_design(
            model_chunk,
            design_spec,
        )

        sqrt_w = np.sqrt(
            w_chunk
        )

        X_weighted = X_chunk.multiply(
            sqrt_w[:, None]
        )

        y_weighted = (
            y_chunk
            * sqrt_w
        )

        xtwx += (
            X_weighted.T
            @ X_weighted
        ).toarray()

        xtwy += np.asarray(
            X_weighted.T
            @ y_weighted
        ).ravel()

        n_control_rows_used += len(
            y_chunk
        )

    if n_control_rows_used == 0:
        raise ValueError(
            "No never-treated observations available for outcome regression."
        )

    # pinv is robust to a category being absent from a particular bootstrap.
    beta = (
        np.linalg.pinv(
            xtwx,
            rcond=1e-10,
        )
        @ xtwy
    )

    return beta


def predict_outcome_model(
    panel_df: pd.DataFrame,
    beta: np.ndarray,
    design_spec: dict[str, list[str]],
) -> np.ndarray:
    """
    Generate fitted values for all treated and control observations in chunks.
    """
    fitted = np.empty(
        len(panel_df),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(panel_df),
        MODEL_CHUNK_SIZE,
    ):
        stop = min(
            start + MODEL_CHUNK_SIZE,
            len(panel_df),
        )

        chunk = panel_df.iloc[
            start:stop
        ]

        X_chunk = make_sparse_design(
            chunk[
                MODEL_CATEGORICAL_COLUMNS
            ],
            design_spec,
        )

        fitted[
            start:stop
        ] = np.asarray(
            X_chunk @ beta
        ).ravel()

    return fitted


###############################################################################
# LOAD UPDATED STEP 8 DATA
###############################################################################

section("LOADING UPDATED STEP 8 OUTPUTS")

cols_needed = [
    "contract_account_hashed",
    "month_year",
    "kwh",
    "Area_number",
    "tariff_category",
    "PV_kW",
    "total_duration_hr",
    "shs_label_numeric",
    "has_shs",
]

prepaid = pd.read_parquet(
    PREPAID_FILE,
    columns=cols_needed,
)

postpaid = pd.read_parquet(
    POSTPAID_FILE,
    columns=cols_needed,
)

print(
    "Prepaid rows:",
    f"{len(prepaid):,}",
    flush=True,
)

print(
    "Postpaid rows:",
    f"{len(postpaid):,}",
    flush=True,
)

prepaid["month_year"] = pd.to_datetime(
    prepaid["month_year"],
    errors="coerce",
)

postpaid["month_year"] = pd.to_datetime(
    postpaid["month_year"],
    errors="coerce",
)

if prepaid["month_year"].isna().any():
    raise ValueError(
        "Prepaid data contain invalid month_year values."
    )

if postpaid["month_year"].isna().any():
    raise ValueError(
        "Postpaid data contain invalid month_year values."
    )

assert_unique_account_month(
    prepaid,
    "Prepaid",
)

assert_unique_account_month(
    postpaid,
    "Postpaid",
)

df = pd.concat(
    [
        prepaid,
        postpaid,
    ],
    ignore_index=True,
)

assert_unique_account_month(
    df,
    "Combined",
)

print(
    "Combined unique households:",
    f"{df['contract_account_hashed'].nunique():,}",
    flush=True,
)


###############################################################################
# INITIAL CLEANING
###############################################################################

section("INITIAL CLEANING")

df = df.loc[
    df["month_year"].le(
        END_DATE
    )
].copy()

df["month_year"] = (
    df["month_year"]
    .dt.to_period("M")
)

non_shs_positive_capacity = (
    df["shs_label_numeric"].eq(0)
    & df["PV_kW"].fillna(0).gt(0)
)

print(
    "Non-SHS rows with positive PV_kW:",
    f"{non_shs_positive_capacity.sum():,}",
    flush=True,
)

if non_shs_positive_capacity.any():
    raise ValueError(
        "Some non-SHS observations have positive PV capacity."
    )

shs_missing_capacity = (
    df["shs_label_numeric"].eq(1)
    & df["PV_kW"].isna()
)

print(
    "SHS rows with missing PV_kW:",
    f"{shs_missing_capacity.sum():,}",
    flush=True,
)

if shs_missing_capacity.any():
    raise ValueError(
        "Some SHS observations have missing PV capacity."
    )

df["PV_kW"] = (
    df["PV_kW"]
    .fillna(0)
)

# No household-month aggregation here. Step 8 is already unique.
assert_unique_account_month(
    df,
    "Cleaned combined",
)

print(
    "Panel rows:",
    f"{len(df):,}",
    flush=True,
)

print(
    "Unique households:",
    f"{df['contract_account_hashed'].nunique():,}",
    flush=True,
)


###############################################################################
# TREATMENT COHORTS
###############################################################################

section("DEFINING TREATMENT COHORTS")

# This uses corrected/backfilled Step 8 timing.
# It is the first SHS month in the constructed treatment panel, not necessarily
# the literal physical installation month.
first_pv = (
    df.loc[
        df["PV_kW"].gt(0)
    ]
    .groupby(
        "contract_account_hashed"
    )["month_year"]
    .min()
)

df["treatment_month"] = (
    df["contract_account_hashed"]
    .map(first_pv)
)

valid_treated_ids = (
    first_pv.loc[
        first_pv.ge(
            TREATMENT_CUTOFF
        )
    ]
    .index
)

df = df.loc[
    df[
        "contract_account_hashed"
    ].isin(
        valid_treated_ids
    )
    | df[
        "treatment_month"
    ].isna()
].copy()

print(
    "Treated households retained:",
    f"{len(valid_treated_ids):,}",
    flush=True,
)

print(
    "Never-treated households retained:",
    f"{df.loc[df['treatment_month'].isna(), 'contract_account_hashed'].nunique():,}",
    flush=True,
)

print(
    "Control sampling:",
    "NONE — all never-treated households are retained.",
    flush=True,
)

print(
    "\nHouseholds per treatment cohort:",
    flush=True,
)

print(
    df.loc[
        df["treatment_month"].notna()
    ]
    .groupby(
        "treatment_month"
    )["contract_account_hashed"]
    .nunique()
    .sort_index(),
    flush=True,
)


###############################################################################
# EVENT-STUDY VARIABLES
###############################################################################

section("CONSTRUCTING EVENT-STUDY VARIABLES")

df["D_it"] = (
    df["treatment_month"].notna()
    & df["month_year"].ge(
        df["treatment_month"]
    )
).astype(
    "int8"
)

df["cohort"] = (
    df["treatment_month"]
)

df["time"] = (
    df["month_year"]
)

df["never_treated"] = (
    df["treatment_month"].isna()
)

df["event_time"] = (
    df["time"]
    - df["cohort"]
).apply(
    lambda value: (
        value.n
        if pd.notnull(value)
        else np.nan
    )
)

# Fixed treatment dose based on each household's median positive PV capacity
# in the corrected/backfilled Step-8 panel.

household_pv_capacity = (
    df.loc[
        df["PV_kW"] > 0
    ]
    .groupby(
        "contract_account_hashed"
    )["PV_kW"]
    .median()
)

df["PV_base"] = (
    df["contract_account_hashed"]
    .map(
        household_pv_capacity
    )
    .fillna(0)
)

# Exact non-overlapping dose definitions:
#   0: PV < 1 kW
#   1: 1 <= PV < 2 kW
#   2: 2 <= PV < 3 kW
#   3: 3 <= PV <= 5 kW
#   4: PV > 5 kW
#
# Never-treated households (PV_base == 0) are not assigned a treatment dose.
treated = (
    df["treatment_month"].notna()
)

df["dose_group"] = np.nan

df.loc[
    treated
    & df["PV_base"].lt(1),
    "dose_group",
] = 0

df.loc[
    treated
    & df["PV_base"].ge(1)
    & df["PV_base"].lt(2),
    "dose_group",
] = 1

df.loc[
    treated
    & df["PV_base"].ge(2)
    & df["PV_base"].lt(3),
    "dose_group",
] = 2

df.loc[
    treated
    & df["PV_base"].ge(3)
    & df["PV_base"].le(5),
    "dose_group",
] = 3

df.loc[
    treated
    & df["PV_base"].gt(5),
    "dose_group",
] = 4

missing_treated_dose = (
    treated
    & df["dose_group"].isna()
)

print(
    "Treated rows with missing dose group:",
    f"{missing_treated_dose.sum():,}",
    flush=True,
)

if missing_treated_dose.any():
    raise ValueError(
        "Some treated observations could not be assigned a dose group."
    )

print(
    "\nTreated households per dose group:",
    flush=True,
)

print(
    df.loc[
        treated
    ]
    .groupby(
        "dose_group"
    )["contract_account_hashed"]
    .nunique()
    .reset_index(
        name="unique_contracts"
    )
    .sort_values(
        "dose_group"
    )
    .to_string(
        index=False
    ),
    flush=True,
)

nonpositive_kwh = int(
    df["kwh"].le(0).sum()
)

missing_kwh = int(
    df["kwh"].isna().sum()
)

print(
    "Rows with missing kWh excluded from log outcome:",
    f"{missing_kwh:,}",
    flush=True,
)

print(
    "Rows with kWh <= 0 excluded from log outcome:",
    f"{nonpositive_kwh:,}",
    flush=True,
)

df = df.loc[
    df["kwh"].notna()
    & df["kwh"].gt(0)
].copy()

df["y"] = np.log(
    df["kwh"]
)


###############################################################################
# RESTRICT EVENT WINDOW
###############################################################################

section("RESTRICTING EVENT WINDOW")

df = df.loc[
    df["never_treated"]
    | df["event_time"].between(
        EVENT_MIN,
        EVENT_MAX,
        inclusive="both",
    )
].copy()

df.reset_index(
    drop=True,
    inplace=True,
)

print(
    "Rows after event-window restriction:",
    f"{len(df):,}",
    flush=True,
)

print(
    "Unique never-treated households in estimation panel:",
    f"{df.loc[df['never_treated'], 'contract_account_hashed'].nunique():,}",
    flush=True,
)

print(
    "Unique treated households in estimation panel:",
    f"{df.loc[~df['never_treated'], 'contract_account_hashed'].nunique():,}",
    flush=True,
)


###############################################################################
# HOUSEHOLD INTEGER CODES FOR WEIGHTED CLUSTER BOOTSTRAP
###############################################################################

section("CREATING HOUSEHOLD CODES")

household_codes, household_values = pd.factorize(
    df["contract_account_hashed"],
    sort=False,
)

df["hh_code"] = household_codes.astype(
    np.int32
)

n_households = len(
    household_values
)

print(
    "Unique households coded:",
    f"{n_households:,}",
    flush=True,
)


###############################################################################
# PREPARE FIRST-STAGE MODEL CATEGORIES
###############################################################################

section("PREPARING OUTCOME-REGRESSION DESIGN")

df["Area_number_model"] = (
    df["Area_number"]
    .astype("string")
    .fillna("__MISSING__")
)

df["tariff_model"] = (
    df["tariff_category"]
    .astype("string")
    .fillna("__MISSING__")
)

df["month_model"] = (
    df["month_year"]
    .astype(str)
)

design_spec = make_design_spec(
    df
)

print(
    "Outcome model design columns:",
    f"{design_column_count(design_spec):,}",
    flush=True,
)

for column in MODEL_CATEGORICAL_COLUMNS:
    print(
        f"{column} levels:",
        len(
            design_spec[column]
        ),
        flush=True,
    )


###############################################################################
# ATT(g,t,d) ESTIMATOR
###############################################################################

def compute_event_att(
    panel_df: pd.DataFrame,
    adjusted_outcome: np.ndarray,
    household_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute cohort-time-dose adjusted DiD estimates.

    adjusted_outcome is:
        y_it - mu0_hat(X_it)

    Baseline is always g-1.

    Negative event times are retained as pre-treatment/placebo estimates.
    Event time -1 is the baseline and is therefore omitted.
    """
    if len(adjusted_outcome) != len(
        panel_df
    ):
        raise ValueError(
            "adjusted_outcome length does not match panel rows."
        )

    duplicate_keys = int(
        panel_df.duplicated(
            subset=[
                "contract_account_hashed",
                "time",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_keys > 0:
        raise ValueError(
            "ATT input is not unique by household-time."
        )

    # MultiIndex lookup avoids constructing two very large Python dictionaries.
    adjusted_series = pd.Series(
        adjusted_outcome,
        index=pd.MultiIndex.from_arrays(
            [
                panel_df[
                    "contract_account_hashed"
                ].to_numpy(),
                panel_df[
                    "time"
                ].to_numpy(),
            ],
            names=[
                "contract_account_hashed",
                "time",
            ],
        ),
    )

    if household_weights is None:
        household_weights = np.ones(
            n_households,
            dtype=np.float64,
        )

    min_time = panel_df["time"].min()
    max_time = panel_df["time"].max()

    att_list = []

    treated_cells = (
        panel_df.loc[
            panel_df["cohort"].notna()
        ]
        .groupby(
            [
                "cohort",
                "time",
            ],
            sort=False,
        )
    )

    # Pre-group never-treated current-period row indices for faster reuse.
    control_indices_by_time = (
        panel_df.loc[
            panel_df["never_treated"]
            & panel_df["PV_base"].eq(0)
        ]
        .groupby(
            "time",
            sort=False,
        )
        .indices
    )

    for (
        cohort,
        current_time,
    ), df_gt in treated_cells:

        event_time = (
            current_time
            - cohort
        ).n

        if event_time < EVENT_MIN:
            continue

        if event_time > EVENT_MAX:
            continue

        # g-1 is the reference period.
        if event_time == -1:
            continue

        base_time = (
            cohort
            - 1
        )

        if not (
            min_time
            <= base_time
            <= max_time
        ):
            continue

        control_index = (
            control_indices_by_time.get(
                current_time
            )
        )

        if control_index is None:
            continue

        controls = panel_df.iloc[
            control_index
        ].copy()

        control_current = adjusted_outcome[
            control_index
        ]

        control_base_index = pd.MultiIndex.from_arrays(
            [
                controls[
                    "contract_account_hashed"
                ].to_numpy(),
                np.repeat(
                    base_time,
                    len(controls),
                ),
            ]
        )

        control_base = (
            adjusted_series
            .reindex(
                control_base_index
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        control_change = (
            control_current
            - control_base
        )

        control_weights = household_weights[
            controls[
                "hh_code"
            ].to_numpy(
                dtype=np.int64
            )
        ]

        control_mean = weighted_mean(
            control_change,
            control_weights,
        )

        if not np.isfinite(
            control_mean
        ):
            continue

        for dose in (
            df_gt[
                "dose_group"
            ]
            .dropna()
            .unique()
        ):

            treated_rows = df_gt.loc[
                df_gt[
                    "dose_group"
                ].eq(dose)
            ]

            if treated_rows.empty:
                continue

            treated_index = (
                treated_rows.index.to_numpy(
                    dtype=np.int64
                )
            )

            treated_current = adjusted_outcome[
                treated_index
            ]

            treated_base_index = pd.MultiIndex.from_arrays(
                [
                    treated_rows[
                        "contract_account_hashed"
                    ].to_numpy(),
                    np.repeat(
                        base_time,
                        len(treated_rows),
                    ),
                ]
            )

            treated_base = (
                adjusted_series
                .reindex(
                    treated_base_index
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            treated_change = (
                treated_current
                - treated_base
            )

            treated_weights = household_weights[
                treated_rows[
                    "hh_code"
                ].to_numpy(
                    dtype=np.int64
                )
            ]

            treated_mean = weighted_mean(
                treated_change,
                treated_weights,
            )

            if not np.isfinite(
                treated_mean
            ):
                continue

            n_treated_weight = float(
                treated_weights[
                    (
                        np.isfinite(
                            treated_change
                        )
                        & (
                            treated_weights
                            > 0
                        )
                    )
                ].sum()
            )

            if n_treated_weight <= 0:
                continue

            att_gtd = (
                treated_mean
                - control_mean
            )

            att_list.append(
                {
                    "cohort": cohort,
                    "time": current_time,
                    "dose_group": int(dose),
                    "event_time": int(event_time),
                    "att": float(att_gtd),
                    "n_treated_weight": n_treated_weight,
                }
            )

    att_df = pd.DataFrame(
        att_list
    )

    if att_df.empty:
        raise ValueError(
            "No ATT(g,t,d) cells were estimable."
        )

    rows = []

    for (
        event_time,
        dose_group,
    ), group in att_df.groupby(
        [
            "event_time",
            "dose_group",
        ],
        sort=True,
    ):
        weights = group[
            "n_treated_weight"
        ].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "event_time": int(
                    event_time
                ),
                "dose_group": int(
                    dose_group
                ),
                "att": float(
                    np.average(
                        group[
                            "att"
                        ].to_numpy(
                            dtype=np.float64
                        ),
                        weights=weights,
                    )
                ),
                "n_treated_weight": float(
                    weights.sum()
                ),
                "n_cells": int(
                    len(group)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


###############################################################################
# BASE OUTCOME MODEL + BASE ATT
###############################################################################

section("ESTIMATING BASE OUTCOME MODEL")

base_beta = fit_outcome_model(
    df,
    design_spec,
    household_weights=None,
)

base_fitted = predict_outcome_model(
    df,
    base_beta,
    design_spec,
)

base_adjusted_outcome = (
    df["y"].to_numpy(
        dtype=np.float64
    )
    - base_fitted
)

print(
    "Missing adjusted outcomes:",
    f"{np.isnan(base_adjusted_outcome).sum():,}",
    flush=True,
)

section("COMPUTING BASE ATT")

event_att = compute_event_att(
    df,
    base_adjusted_outcome,
    household_weights=None,
)

print(
    event_att.head(40).to_string(
        index=False
    ),
    flush=True,
)


###############################################################################
# HOUSEHOLD-CLUSTER BOOTSTRAP — CHECKPOINTED AND RESUMABLE
###############################################################################

section("BOOTSTRAPPING STANDARD ERRORS")

# Each bootstrap iteration is deterministic from its own seed:
# RANDOM_SEED + bootstrap_iteration.
#
# This means a restarted job can skip already-completed iterations without
# needing to reconstruct the state of a single sequential random-number stream.

def bootstrap_checkpoint_path(
    bootstrap_iteration: int,
) -> str:
    return os.path.join(
        BOOTSTRAP_CHECKPOINT_DIR,
        f"boot_iter_{bootstrap_iteration:04d}_09182026_max5.parquet",
    )


def bootstrap_iteration_is_complete(
    bootstrap_iteration: int,
) -> bool:
    checkpoint_file = bootstrap_checkpoint_path(
        bootstrap_iteration
    )

    if not os.path.exists(
        checkpoint_file
    ):
        return False

    # Make sure an interrupted write did not leave an unreadable file.
    try:
        check = pd.read_parquet(
            checkpoint_file,
            columns=[
                "event_time",
                "dose_group",
                "att",
                "boot_iter",
            ],
        )

        return (
            len(check) > 0
            and check[
                "boot_iter"
            ].eq(
                bootstrap_iteration
            ).all()
        )

    except Exception as exc:
        print(
            "Unreadable bootstrap checkpoint will be recomputed:",
            checkpoint_file,
            repr(exc),
            flush=True,
        )

        return False


already_complete = [
    bootstrap_iteration
    for bootstrap_iteration in range(
        BOOTSTRAP_ITERATIONS
    )
    if bootstrap_iteration_is_complete(
        bootstrap_iteration
    )
]

print(
    "Previously completed bootstrap iterations:",
    f"{len(already_complete):,}",
    "of",
    BOOTSTRAP_ITERATIONS,
    flush=True,
)

for bootstrap_iteration in range(
    BOOTSTRAP_ITERATIONS
):

    checkpoint_file = bootstrap_checkpoint_path(
        bootstrap_iteration
    )

    if bootstrap_iteration_is_complete(
        bootstrap_iteration
    ):
        print(
            "Skipping completed bootstrap iteration",
            f"{bootstrap_iteration + 1}/{BOOTSTRAP_ITERATIONS}",
            flush=True,
        )
        continue

    print(
        "Bootstrap iteration",
        f"{bootstrap_iteration + 1}/{BOOTSTRAP_ITERATIONS}",
        flush=True,
    )

    # Use an iteration-specific RNG so restart/resume is exactly reproducible.
    bootstrap_rng = np.random.default_rng(
        RANDOM_SEED
        + bootstrap_iteration
    )

    # Draw HOUSEHOLDS with replacement, retaining their complete panels.
    sampled_household_codes = bootstrap_rng.integers(
        low=0,
        high=n_households,
        size=n_households,
    )

    household_weights = np.bincount(
        sampled_household_codes,
        minlength=n_households,
    ).astype(
        np.float64
    )

    # Re-estimate the first-stage outcome regression inside every bootstrap draw.
    bootstrap_beta = fit_outcome_model(
        df,
        design_spec,
        household_weights=household_weights,
    )

    bootstrap_fitted = predict_outcome_model(
        df,
        bootstrap_beta,
        design_spec,
    )

    bootstrap_adjusted_outcome = (
        df["y"].to_numpy(
            dtype=np.float64
        )
        - bootstrap_fitted
    )

    # Re-estimate ATT(g,t,d), weighting households by bootstrap multiplicity.
    event_att_b = compute_event_att(
        df,
        bootstrap_adjusted_outcome,
        household_weights=household_weights,
    )

    event_att_b[
        "boot_iter"
    ] = bootstrap_iteration

    event_att_b = event_att_b[
        [
            "event_time",
            "dose_group",
            "att",
            "boot_iter",
        ]
    ].copy()

    # Write atomically: first a temporary file, then rename it.
    temporary_file = (
        checkpoint_file
        + ".tmp.parquet"
    )

    event_att_b.to_parquet(
        temporary_file,
        index=False,
    )

    os.replace(
        temporary_file,
        checkpoint_file,
    )

    print(
        "Saved bootstrap checkpoint:",
        checkpoint_file,
        flush=True,
    )


###############################################################################
# COMBINE AND SAVE ALL BOOTSTRAP DRAWS
###############################################################################

section("COMBINING BOOTSTRAP DRAWS")

missing_iterations = [
    bootstrap_iteration
    for bootstrap_iteration in range(
        BOOTSTRAP_ITERATIONS
    )
    if not bootstrap_iteration_is_complete(
        bootstrap_iteration
    )
]

if missing_iterations:
    raise RuntimeError(
        "Bootstrap did not finish all iterations. "
        f"Missing iterations: {missing_iterations}"
    )

boot_results = []

for bootstrap_iteration in range(
    BOOTSTRAP_ITERATIONS
):
    checkpoint_file = bootstrap_checkpoint_path(
        bootstrap_iteration
    )

    boot_results.append(
        pd.read_parquet(
            checkpoint_file
        )
    )

boot_df = pd.concat(
    boot_results,
    ignore_index=True,
)

expected_boot_iterations = set(
    range(
        BOOTSTRAP_ITERATIONS
    )
)

observed_boot_iterations = set(
    boot_df[
        "boot_iter"
    ].unique()
)

if observed_boot_iterations != expected_boot_iterations:
    raise RuntimeError(
        "Combined bootstrap file does not contain exactly the expected "
        "bootstrap iterations."
    )

print(
    "Combined bootstrap rows:",
    f"{len(boot_df):,}",
    flush=True,
)

print(
    "Combined bootstrap iterations:",
    f"{boot_df['boot_iter'].nunique():,}",
    flush=True,
)

# SAVE THE FULL BOOTSTRAP DISTRIBUTION.
# This is what allows later inference/plots/equivalence checks WITHOUT
# rerunning the seven-day event-study bootstrap.
boot_df.to_parquet(
    BOOTSTRAP_DRAWS_FILE,
    index=False,
)

print(
    "Saved complete bootstrap draws:",
    BOOTSTRAP_DRAWS_FILE,
    flush=True,
)


###############################################################################
# STANDARD ERRORS / CONFIDENCE INTERVALS
###############################################################################

se_df = (
    boot_df
    .groupby(
        [
            "event_time",
            "dose_group",
        ]
    )["att"]
    .std()
    .reset_index(
        name="se"
    )
)

event_att = event_att.merge(
    se_df,
    on=[
        "event_time",
        "dose_group",
    ],
    how="left",
    validate="one_to_one",
)

event_att["ci_low"] = (
    event_att["att"]
    - 1.96
    * event_att["se"]
)

event_att["ci_high"] = (
    event_att["att"]
    + 1.96
    * event_att["se"]
)

print(
    "\nMissing ATT/SE values:",
    flush=True,
)

print(
    event_att[
        [
            "att",
            "se",
        ]
    ].isna().sum(),
    flush=True,
)

print(
    "Final event-study rows:",
    f"{len(event_att):,}",
    flush=True,
)


###############################################################################
# DETTE-SCHUMANN MAXIMUM PRE-TREND EQUIVALENCE TEST
###############################################################################

section("TESTING PRE-TREND EQUIVALENCE")

# We test, separately by PV-capacity group:
#
#   H0: ||beta_pre||_infinity >= delta
#   H1: ||beta_pre||_infinity <  delta
#
# where beta_pre contains event-study placebo coefficients for event times
# EVENT_MIN,...,-2. Event time -1 is the omitted reference period.
#
# The implementation uses the general-estimator Gaussian approximation discussed
# in Remark 4.2 of Dette & Schumann (2024). The covariance matrix of the
# pre-treatment ATT vector is estimated from the HOUSEHOLD-CLUSTER bootstrap
# draws generated above.
#
# Because our ATT estimator is not their original TWFE regression estimator,
# report this in the paper as an application/extension of their maximum-norm
# equivalence approach, rather than implying that this is their exact original
# regression algorithm.


def make_positive_semidefinite(
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = (
        covariance
        + covariance.T
    ) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    floor = max(
        float(
            np.max(
                np.abs(
                    eigenvalues
                )
            )
        )
        * 1e-10,
        1e-12,
    )

    eigenvalues = np.clip(
        eigenvalues,
        floor,
        None,
    )

    return (
        eigenvectors
        @ np.diag(
            eigenvalues
        )
        @ eigenvectors.T
    )


def constrained_boundary_estimate(
    beta_hat: np.ndarray,
    covariance: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Approximate the constrained estimator under:

        max_j |beta_j| = delta.

    When beta_hat already satisfies H0, Dette-Schumann use the unrestricted
    estimate as the bootstrap center. Otherwise we project beta_hat onto the
    max-norm boundary using the covariance-weighted (Mahalanobis) distance.
    """
    if np.max(
        np.abs(
            beta_hat
        )
    ) >= delta:
        return beta_hat.copy()

    precision = np.linalg.pinv(
        covariance
    )

    def objective(
        candidate: np.ndarray,
    ) -> float:
        difference = (
            candidate
            - beta_hat
        )

        return float(
            difference.T
            @ precision
            @ difference
        )

    bounds = [
        (
            -delta,
            delta,
        )
        for _ in range(
            len(
                beta_hat
            )
        )
    ]

    best_candidate = None
    best_objective = np.inf

    # max |beta| = delta is a union of 2*p boundary faces.
    for coordinate in range(
        len(
            beta_hat
        )
    ):
        for sign in (
            -1.0,
            1.0,
        ):
            start = np.clip(
                beta_hat,
                -delta,
                delta,
            )

            start[
                coordinate
            ] = (
                sign
                * delta
            )

            constraints = [
                {
                    "type": "eq",
                    "fun": (
                        lambda candidate,
                        coordinate=coordinate,
                        sign=sign: (
                            candidate[
                                coordinate
                            ]
                            - sign
                            * delta
                        )
                    ),
                }
            ]

            result = optimize.minimize(
                objective,
                x0=start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": 500,
                    "ftol": 1e-12,
                    "disp": False,
                },
            )

            if (
                result.success
                and result.fun
                < best_objective
            ):
                best_candidate = (
                    result.x.copy()
                )

                best_objective = float(
                    result.fun
                )

    if best_candidate is None:
        raise RuntimeError(
            "Could not obtain constrained maximum-norm equivalence estimate."
        )

    return best_candidate


def max_equivalence_test(
    beta_hat: np.ndarray,
    covariance: np.ndarray,
    delta: float,
    standard_normal_draws: np.ndarray,
) -> dict:
    """
    Dette-Schumann-style maximum-norm equivalence test.

    Reject H0 of a non-negligible pre-trend if:

        ||beta_hat||_infinity < Q_alpha^*

    where Q_alpha^* is the alpha quantile of the simulated maximum absolute
    placebo coefficient under the constrained null.
    """
    beta_center = constrained_boundary_estimate(
        beta_hat,
        covariance,
        delta,
    )

    covariance = make_positive_semidefinite(
        covariance
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    covariance_sqrt = (
        eigenvectors
        @ np.diag(
            np.sqrt(
                eigenvalues
            )
        )
        @ eigenvectors.T
    )

    simulated_beta = (
        beta_center[
            None,
            :
        ]
        + standard_normal_draws
        @ covariance_sqrt.T
    )

    simulated_maximum = np.max(
        np.abs(
            simulated_beta
        ),
        axis=1,
    )

    q_alpha = float(
        np.quantile(
            simulated_maximum,
            EQUIVALENCE_ALPHA,
        )
    )

    observed_maximum = float(
        np.max(
            np.abs(
                beta_hat
            )
        )
    )

    reject_non_equivalence = (
        observed_maximum
        < q_alpha
    )

    # Monte-Carlo analog of the lower-tail rejection probability.
    p_value = float(
        np.mean(
            simulated_maximum
            <= observed_maximum
        )
    )

    return {
        "delta": float(
            delta
        ),
        "observed_max_abs_pretrend": observed_maximum,
        "null_q_alpha": q_alpha,
        "equivalent": bool(
            reject_non_equivalence
        ),
        "p_value": p_value,
    }


def smallest_supported_equivalence_threshold(
    beta_hat: np.ndarray,
    covariance: np.ndarray,
    standard_normal_draws: np.ndarray,
) -> float:
    """
    Find the smallest delta at which H0 of non-equivalence is rejected.

    Uses a deterministic binary search with common Gaussian draws.
    """
    observed_maximum = float(
        np.max(
            np.abs(
                beta_hat
            )
        )
    )

    lower = max(
        observed_maximum,
        1e-8,
    )

    upper = max(
        0.02,
        lower
        * 1.5,
    )

    # Expand until equivalence is supported.
    for _ in range(
        30
    ):
        result = max_equivalence_test(
            beta_hat,
            covariance,
            upper,
            standard_normal_draws,
        )

        if result[
            "equivalent"
        ]:
            break

        upper *= 1.5

    else:
        return np.nan

    # Refine boundary.
    for _ in range(
        30
    ):
        midpoint = (
            lower
            + upper
        ) / 2

        result = max_equivalence_test(
            beta_hat,
            covariance,
            midpoint,
            standard_normal_draws,
        )

        if result[
            "equivalent"
        ]:
            upper = midpoint
        else:
            lower = midpoint

    return float(
        upper
    )


pre_event_times = [
    event_time
    for event_time in range(
        EVENT_MIN,
        -1,
    )
    if event_time != -1
]

dose_labels = {
    0: "<1 kW",
    1: "1–2 kW",
    2: "2–3 kW",
    3: "3–5 kW",
    4: ">5 kW",
}

equivalence_rows = []

for dose_group in sorted(
    event_att[
        "dose_group"
    ].dropna().unique()
):
    dose_group = int(
        dose_group
    )

    point_pre = (
        event_att.loc[
            (
                event_att[
                    "dose_group"
                ].eq(
                    dose_group
                )
            )
            & (
                event_att[
                    "event_time"
                ].isin(
                    pre_event_times
                )
            ),
            [
                "event_time",
                "att",
            ],
        ]
        .sort_values(
            "event_time"
        )
    )

    if point_pre.empty:
        print(
            "No pre-treatment estimates for dose group:",
            dose_group,
            flush=True,
        )
        continue

    usable_event_times = (
        point_pre[
            "event_time"
        ]
        .astype(int)
        .tolist()
    )

    beta_hat = (
        point_pre[
            "att"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    bootstrap_pre = (
        boot_df.loc[
            (
                boot_df[
                    "dose_group"
                ].eq(
                    dose_group
                )
            )
            & (
                boot_df[
                    "event_time"
                ].isin(
                    usable_event_times
                )
            ),
            [
                "boot_iter",
                "event_time",
                "att",
            ],
        ]
        .pivot(
            index="boot_iter",
            columns="event_time",
            values="att",
        )
        .reindex(
            columns=usable_event_times
        )
        .dropna()
    )

    if len(
        bootstrap_pre
    ) < 30:
        print(
            "Too few complete bootstrap draws for equivalence test, dose:",
            dose_group,
            "draws:",
            len(
                bootstrap_pre
            ),
            flush=True,
        )
        continue

    covariance = np.cov(
        bootstrap_pre.to_numpy(
            dtype=np.float64
        ),
        rowvar=False,
        ddof=1,
    )

    covariance = np.atleast_2d(
        covariance
    )

    covariance = make_positive_semidefinite(
        covariance
    )

    simulation_rng = np.random.default_rng(
        RANDOM_SEED
        + 100_000
        + dose_group
    )

    standard_normal_draws = simulation_rng.standard_normal(
        size=(
            EQUIVALENCE_SIM_DRAWS,
            len(
                beta_hat
            ),
        )
    )

    smallest_delta = smallest_supported_equivalence_threshold(
        beta_hat,
        covariance,
        standard_normal_draws,
    )

    observed_maximum = float(
        np.max(
            np.abs(
                beta_hat
            )
        )
    )

    base_row = {
        "dose_group": dose_group,
        "dose_label": dose_labels.get(
            dose_group,
            str(
                dose_group
            ),
        ),
        "n_pre_periods": len(
            beta_hat
        ),
        "n_complete_bootstrap_draws": len(
            bootstrap_pre
        ),
        "observed_max_abs_pretrend_log_points": observed_maximum,
        "observed_max_abs_pretrend_approx_pct": (
            100
            * (
                np.exp(
                    observed_maximum
                )
                - 1
            )
        ),
        "smallest_supported_delta_log_points": smallest_delta,
        "smallest_supported_delta_approx_pct": (
            100
            * (
                np.exp(
                    smallest_delta
                )
                - 1
            )
            if np.isfinite(
                smallest_delta
            )
            else np.nan
        ),
    }

    # Evaluate several interpretable candidate thresholds too.
    for delta in EQUIVALENCE_THRESHOLDS:
        test_result = max_equivalence_test(
            beta_hat,
            covariance,
            delta,
            standard_normal_draws,
        )

        threshold_name = (
            str(
                delta
            )
            .replace(
                ".",
                "p"
            )
        )

        base_row[
            f"equivalent_at_delta_{threshold_name}"
        ] = test_result[
            "equivalent"
        ]

        base_row[
            f"p_value_delta_{threshold_name}"
        ] = test_result[
            "p_value"
        ]

    equivalence_rows.append(
        base_row
    )


equivalence_df = pd.DataFrame(
    equivalence_rows
)

equivalence_df.to_csv(
    EQUIVALENCE_OUTPUT_FILE,
    index=False,
)

print(
    "\nMaximum pre-trend equivalence results:",
    flush=True,
)

print(
    equivalence_df.to_string(
        index=False
    ),
    flush=True,
)

print(
    "\nSaved pre-trend equivalence results:",
    EQUIVALENCE_OUTPUT_FILE,
    flush=True,
)


###############################################################################
# PRE-TREND EQUIVALENCE DIAGNOSTIC PLOT
###############################################################################

if not equivalence_df.empty:
    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    x = np.arange(
        len(
            equivalence_df
        )
    )

    ax.bar(
        x,
        equivalence_df[
            "observed_max_abs_pretrend_log_points"
        ],
        label="Observed maximum absolute pre-trend",
    )

    ax.scatter(
        x,
        equivalence_df[
            "smallest_supported_delta_log_points"
        ],
        marker="D",
        s=65,
        label="Smallest supported equivalence bound",
        zorder=3,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        equivalence_df[
            "dose_label"
        ]
    )

    ax.set_ylabel(
        "Absolute deviation in log monthly consumption"
    )

    ax.set_xlabel(
        "PV capacity group"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    equivalence_plot = os.path.join(
        PLOT_DIR,
        "pretrend_equivalence_max_09182026_max5.png",
    )

    fig.savefig(
        equivalence_plot,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved pre-trend equivalence plot:",
        equivalence_plot,
        flush=True,
    )


###############################################################################
# SAVE ESTIMATES
###############################################################################

section("SAVING EVENT-STUDY ESTIMATES")

event_output = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_estimates_09182026_max5.csv",
)

event_att.to_csv(
    event_output,
    index=False,
)

print(
    "Saved event-study estimates:",
    event_output,
    flush=True,
)


###############################################################################
# PLOTS
###############################################################################

section("PLOTTING EVENT-STUDY RESULTS")

dose_labels = {
    0: "<1 kW",
    1: "1–2 kW",
    2: "2–3 kW",
    3: "3–5 kW",
    4: ">5 kW",
}


# Plot 1: lines + 95% confidence bands
fig, ax = plt.subplots(
    figsize=(10, 6)
)

for dose in sorted(
    event_att[
        "dose_group"
    ].dropna().unique()
):
    tmp = (
        event_att.loc[
            event_att[
                "dose_group"
            ].eq(dose)
        ]
        .sort_values(
            "event_time"
        )
    )

    ax.plot(
        tmp["event_time"],
        tmp["att"],
        label=dose_labels.get(
            int(dose),
            str(dose),
        ),
    )

    ax.fill_between(
        tmp["event_time"],
        tmp["ci_low"],
        tmp["ci_high"],
        alpha=0.2,
    )

ax.axvline(
    -1,
    linestyle="--",
    color="black",
)

ax.axhline(
    0,
    linestyle="--",
    color="black",
)

ax.set_xlabel(
    "Event time (months)"
)

ax.set_ylabel(
    "ATT on log monthly electricity consumption"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.3
)

fig.tight_layout()

plot1 = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_lines_09182026_max5.png",
)

fig.savefig(
    plot1,
    dpi=600,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "Saved:",
    plot1,
    flush=True,
)


# Plot 2: jittered point estimates + 95% CI
fig, ax = plt.subplots(
    figsize=(10, 6)
)

dose_groups = sorted(
    event_att[
        "dose_group"
    ].dropna().unique()
)

colors = plt.cm.tab10.colors

offsets = np.linspace(
    -0.15,
    0.15,
    len(dose_groups),
)

for index, dose in enumerate(
    dose_groups
):
    tmp = (
        event_att.loc[
            event_att[
                "dose_group"
            ].eq(dose)
        ]
        .sort_values(
            "event_time"
        )
    )

    color = colors[
        index
        % len(colors)
    ]

    x_jittered = (
        tmp["event_time"]
        + offsets[index]
    )

    ax.scatter(
        x_jittered,
        tmp["att"],
        s=40,
        color=color,
        label=dose_labels.get(
            int(dose),
            str(dose),
        ),
        zorder=3,
    )

    ax.plot(
        x_jittered,
        tmp["att"],
        linewidth=1,
        alpha=0.7,
        color=color,
        zorder=2,
    )

    ax.errorbar(
        x_jittered,
        tmp["att"],
        yerr=1.96
        * tmp["se"],
        fmt="none",
        ecolor=color,
        elinewidth=1,
        capsize=3,
        alpha=0.9,
        zorder=1,
    )

ax.axvline(
    -1,
    linestyle="--",
    color="black",
    linewidth=1,
)

ax.axhline(
    0,
    linestyle="--",
    color="black",
    linewidth=1,
)

ax.set_xlabel(
    "Event time (months)"
)

ax.set_ylabel(
    "ATT on log monthly electricity consumption"
)

ax.legend(
    frameon=False
)

ax.grid(
    False
)

ax.set_xticks(
    sorted(
        event_att[
            "event_time"
        ].unique()
    )
)

fig.tight_layout()

plot2 = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_points_09182026_max5.png",
)

fig.savefig(
    plot2,
    dpi=600,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "Saved:",
    plot2,
    flush=True,
)


###############################################################################
# COMPLETE
###############################################################################

section("COMPLETE")

print(
    "Updated all-household event-study workflow completed successfully.",
    flush=True,
)