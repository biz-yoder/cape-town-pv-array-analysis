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
    "/shared/gaia-homes/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/prepaid_cleaned_vector_noagg.parquet"
)
POSTPAID_FILE = (
    "/shared/gaia-homes/ey53/vscode-server-backup/CapeTown_Workflow/"
    "8_out_09132026_max5/postpaid_cleaned_vector_noagg.parquet"
)
PLOT_DIR = (
    "/shared/gaia-homes/ey53/vscode-server-backup/CapeTown_Workflow/"
    "10_Plots_10272026_timing_shift_sensitivity"
)
END_DATE = pd.Timestamp("2023-12-31")
TREATMENT_CUTOFF = pd.Period("2018-06", freq="M")
BOOTSTRAP_ITERATIONS = 200
RANDOM_SEED = 42
EVENT_MIN = -12
EVENT_MAX = 36
SHIFTS = [-6, -3, 0, 3, 6]
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
        EVENT_MIN - 6,
        EVENT_MAX + 6,
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
    diagnostic_frames: list | None = None,
    shift_months: int = 0,
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
            # Optional audit: exact treated pairs used by this ATT(g,t,d).
            # Limit collection to the common ORIGINAL event months +6..+12.
            # The original cohort and household code remain fixed across shifts.
            original_event_time = int(event_time + shift_months)
            if (diagnostic_frames is not None
                    and 6 <= original_event_time <= 12):
                eligible = (
                    np.isfinite(treated_change)
                    & np.isfinite(treated_weights)
                    & (treated_weights > 0)
                )
                if eligible.any():
                    diagnostic_frames.append(pd.DataFrame({
                        "shift_months": int(shift_months),
                        "dose_group": int(dose),
                        "original_event_time": original_event_time,
                        "original_cohort": str(cohort - shift_months),
                        "hh_code": treated_rows.loc[eligible, "hh_code"].to_numpy(dtype=np.int32),
                        "household_weight": treated_weights[eligible].astype(np.float64),
                    }))
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
# FIT THE NEVER-TREATED OUTCOME MODEL ONCE
###############################################################################
section("FITTING NEVER-TREATED OUTCOME MODEL")
base_beta = fit_outcome_model(df, design_spec)
df["adjusted_y"] = df["y"].to_numpy(dtype=float) - predict_outcome_model(df, base_beta, design_spec)

###############################################################################
# SHIFT THE EVENT-TIME ORIGIN; KEEP STEP 8 DATA UNCHANGED
###############################################################################
all_results = []
diagnostic_frames = []
for shift in SHIFTS:
    section(f"ESTIMATING EVENT STUDY: shift {shift:+d} months")
    # Positive shift means a later reference installation month.
    # Cohort is a pandas monthly Period; addition shifts calendar months.
    scenario = df.copy()
    treated_mask = scenario["treatment_month"].notna()
    scenario.loc[treated_mask, "cohort"] = scenario.loc[treated_mask, "treatment_month"] + shift
    scenario.loc[treated_mask, "event_time"] = [
        (t - c).n for t, c in zip(
            scenario.loc[treated_mask, "time"],
            scenario.loc[treated_mask, "cohort"]
        )
    ]
    # Same observed households and outcomes; different event-time assignment.
    scenario = scenario.loc[
        scenario["never_treated"] | scenario["event_time"].between(EVENT_MIN, EVENT_MAX)
    ].copy().reset_index(drop=True)
    # The estimator expects a positional adjusted outcome vector.
    att = compute_event_att(
        scenario, scenario["adjusted_y"].to_numpy(dtype=float),
        diagnostic_frames=diagnostic_frames, shift_months=shift,
    )
    att["shift_months"] = shift
    att["scenario"] = f"{shift:+d} months"
    att["pct_change"] = 100 * np.expm1(att["att"])
    all_results.append(att)
    print(att.loc[att["dose_group"].eq(4) & att["event_time"].between(-12, 12),
                  ["event_time", "att", "n_treated_weight"]].to_string(index=False), flush=True)

results = pd.concat(all_results, ignore_index=True)
results.to_csv(os.path.join(PLOT_DIR, "event_study_all_timing_shifts.csv"), index=False)

# Diagnostics: largest absolute pre-period point estimate and mean +6 to +12 ATT.
summary_rows = []
for (dose, shift), part in results.groupby(["dose_group", "shift_months"]):
    pre = part.loc[part["event_time"].between(-12, -2)]
    post = part.loc[part["event_time"].between(6, 12)]
    mean_post = post["att"].mean() if len(post) == 7 else np.nan
    summary_rows.append({
        "dose_group": int(dose), "shift_months": int(shift),
        "max_abs_pretrend_log": pre["att"].abs().max(),
        "n_pre_periods": len(pre),
        "mean_att_months_6_to_12_log": mean_post,
        "mean_att_months_6_to_12_pct": 100 * np.expm1(mean_post),
        "n_post_periods_6_to_12": len(post),
    })
summary = pd.DataFrame(summary_rows).sort_values(["dose_group", "shift_months"])
summary.to_csv(os.path.join(PLOT_DIR, "timing_shift_summary.csv"), index=False)
print("\n>5 kW SUMMARY\n", summary.loc[summary["dose_group"].eq(4)].to_string(index=False), flush=True)

labels = {0:"lt1kW", 1:"1to2kW", 2:"2to3kW", 3:"3to5kW", 4:"gt5kW"}
for dose, name in labels.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    for shift in SHIFTS:
        sub = results.loc[(results["dose_group"] == dose) &
                          (results["shift_months"] == shift) &
                          results["event_time"].between(-12, 18)].sort_values("event_time")
        ax.plot(sub["event_time"], sub["att"], marker=".", label=f"{shift:+d} months")
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Months relative to scenario-specific reference date",
           ylabel="ATT (log monthly grid electricity consumption)",
           title=f"Installation-date sensitivity: {name}")
    ax.legend(title="Reference date shift")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"event_study_timing_shifts_{name}.png"), dpi=200)
    plt.close(fig)

print("\nSaved outputs in:", PLOT_DIR, flush=True)
print("Point estimates only; no bootstrap confidence intervals.", flush=True)


###############################################################################
# COMMON-CALENDAR HOUSEHOLD AND COHORT COMPOSITION AUDIT
###############################################################################
section("AUDITING CONTRIBUTING HOUSEHOLDS AND COHORT WEIGHTS")
if not diagnostic_frames:
    raise RuntimeError("No household diagnostics were collected for original months +6..+12.")
contrib = pd.concat(diagnostic_frames, ignore_index=True)
keys = ["dose_group", "original_event_time", "shift_months"]
if contrib.duplicated(keys + ["hh_code"]).any():
    raise ValueError("Duplicate contributing household within a dose/month/shift cell.")
contrib_path = os.path.join(PLOT_DIR, "timing_shift_contributing_households.parquet")
contrib.to_parquet(contrib_path, index=False)
print("Saved exact treated household pairs:", contrib_path, flush=True)

cohort_keys = keys + ["original_cohort"]
cohort_weights = (contrib.groupby(cohort_keys, observed=True)["household_weight"]
                  .sum().rename("n_treated_weight").reset_index())
cohort_weights["cohort_share"] = (
    cohort_weights["n_treated_weight"] /
    cohort_weights.groupby(keys)["n_treated_weight"].transform("sum")
)
cohort_weights.to_csv(os.path.join(PLOT_DIR, "timing_shift_original_cohort_weights.csv"), index=False)

summary_audit = []
for (dose, month), subset in contrib.groupby(["dose_group", "original_event_time"]):
    base_ids = set(subset.loc[subset["shift_months"].eq(0), "hh_code"])
    base_cohorts = cohort_weights.loc[
        cohort_weights["dose_group"].eq(dose)
        & cohort_weights["original_event_time"].eq(month)
        & cohort_weights["shift_months"].eq(0),
        ["original_cohort", "cohort_share"]
    ].set_index("original_cohort")["cohort_share"]
    for shift in SHIFTS:
        group = subset.loc[subset["shift_months"].eq(shift)]
        ids = set(group["hh_code"])
        both = len(ids & base_ids)
        shares = cohort_weights.loc[
            cohort_weights["dose_group"].eq(dose)
            & cohort_weights["original_event_time"].eq(month)
            & cohort_weights["shift_months"].eq(shift),
            ["original_cohort", "cohort_share"]
        ].set_index("original_cohort")["cohort_share"]
        aligned = pd.concat([base_cohorts.rename("base"), shares.rename("shift")], axis=1).fillna(0)
        summary_audit.append({
            "dose_group": int(dose),
            "original_event_time": int(month),
            "shift_months": int(shift),
            "n_baseline_households": len(base_ids),
            "n_scenario_households": len(ids),
            "n_shared_households": both,
            "pct_baseline_households_retained": 100 * both / len(base_ids) if base_ids else np.nan,
            "pct_scenario_households_in_baseline": 100 * both / len(ids) if ids else np.nan,
            "n_baseline_cohorts": len(base_cohorts),
            "n_scenario_cohorts": len(shares),
            "cohort_weight_total_variation": 0.5 * (aligned["base"] - aligned["shift"]).abs().sum(),
            "max_abs_cohort_share_change": (aligned["base"] - aligned["shift"]).abs().max(),
        })
audit = pd.DataFrame(summary_audit).sort_values(keys)
expected = {(d, m, s) for d in range(5) for m in range(6, 13) for s in SHIFTS}
actual = set(map(tuple, audit[keys].to_numpy()))
if missing := expected - actual:
    raise RuntimeError(f"Missing diagnostic cells: {sorted(missing)}")
audit.to_csv(os.path.join(PLOT_DIR, "timing_shift_household_cohort_audit_by_month.csv"), index=False)
compact = audit.groupby(["dose_group", "shift_months"]).agg(
    min_baseline_retained_pct=("pct_baseline_households_retained", "min"),
    mean_baseline_retained_pct=("pct_baseline_households_retained", "mean"),
    max_cohort_weight_distance=("cohort_weight_total_variation", "max"),
    mean_cohort_weight_distance=("cohort_weight_total_variation", "mean"),
    min_contributing_households=("n_scenario_households", "min"),
    max_contributing_households=("n_scenario_households", "max"),
).reset_index()
compact.to_csv(os.path.join(PLOT_DIR, "timing_shift_household_cohort_audit_summary.csv"), index=False)
print("\nHOUSEHOLD / COHORT AUDIT SUMMARY\n", compact.round(4).to_string(index=False), flush=True)
print("\nAudit outputs saved in:", PLOT_DIR, flush=True)
