import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import EVENT_STUDY_DIR

###############################################################################
# PATHS
###############################################################################

PLOT_DIR = (
    EVENT_STUDY_DIR
)

ESTIMATES_FILE = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_estimates.csv",
)

OUTPUT_PNG = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_five_panels.png",
)
OUTPUT_PDF = os.path.join(
    PLOT_DIR,
    "callaway_dose_response_five_panels.pdf",
)

###############################################################################
# LOAD SAVED ESTIMATES
###############################################################################

event_att = pd.read_csv(ESTIMATES_FILE)
required = {"event_time", "dose_group", "att", "se", "ci_low", "ci_high"}
assert required.issubset(event_att.columns), f"Missing columns: {required - set(event_att.columns)}"
assert set(event_att["dose_group"]) == set(range(5)), "Unexpected dose groups"
assert not event_att.duplicated(["dose_group", "event_time"]).any(), "Duplicate points"
assert event_att[list(required)].notna().all().all(), "Missing plotted values"
assert ((event_att["ci_low"] <= event_att["att"]) & (event_att["att"] <= event_att["ci_high"])).all(), "Invalid CI"
assert ((event_att["ci_low"] - (event_att["att"] - 1.96 * event_att["se"])).abs() < 1e-6).all(), "CI and SE disagree"
print("Verified input:", ESTIMATES_FILE, "rows:", len(event_att))

dose_labels = {
    0: "<1 kW",
    1: "1–2 kW",
    2: "2–3 kW",
    3: "3–5 kW",
    4: ">5 kW",
}

dose_groups = [0, 1, 2, 3, 4]

###############################################################################
# COMMON AXIS LIMITS
###############################################################################

ci_min = event_att["ci_low"].min()
ci_max = event_att["ci_high"].max()

padding = 0.05 * (ci_max - ci_min)

ymin = ci_min - padding
ymax = ci_max + padding

xmin = event_att["event_time"].min()
xmax = event_att["event_time"].max()

x_ticks = [-12, -6, 0, 6, 12, 18, 24, 30, 36]

###############################################################################
# CREATE CENTERED FIVE-PANEL LAYOUT
###############################################################################

fig = plt.figure(figsize=(15, 9))

gs = fig.add_gridspec(
    2,
    6,
    hspace=0.28,
    wspace=0.35,
)

# Top row: three equally sized panels
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0, 4:6], sharex=ax1, sharey=ax1)

# Bottom row: two panels centered underneath
ax4 = fig.add_subplot(gs[1, 1:3], sharex=ax1, sharey=ax1)
ax5 = fig.add_subplot(gs[1, 3:5], sharex=ax1, sharey=ax1)

axes = [ax1, ax2, ax3, ax4, ax5]

###############################################################################
# PLOT EACH DOSE GROUP
###############################################################################

for ax, dose in zip(axes, dose_groups):

    tmp = (
        event_att.loc[
            event_att["dose_group"].eq(dose)
        ]
        .sort_values("event_time")
    )

    # Connect point estimates
    ax.plot(
        tmp["event_time"],
        tmp["att"],
        linewidth=1.2,
        zorder=2,
    )

    # Point estimates
    ax.scatter(
        tmp["event_time"],
        tmp["att"],
        s=35,
        zorder=3,
    )

    # 95% confidence intervals
    ax.errorbar(
        tmp["event_time"],
        tmp["att"],
        yerr=1.96 * tmp["se"],
        fmt="none",
        elinewidth=1,
        capsize=2.5,
        alpha=0.9,
        zorder=1,
    )

    # Reference period: month before treatment
    ax.axvline(
        -1,
        linestyle="--",
        color="black",
        linewidth=0.9,
    )

    # Zero-effect line
    ax.axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=0.9,
    )

    ax.set_title(
        dose_labels[dose],
        fontsize=13,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x_ticks)

    ax.grid(False)

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

###############################################################################
# SHARED LABELS
###############################################################################

fig.supxlabel(
    "Event time (months)",
    fontsize=12,
    y=0.04,
)

fig.supylabel(
    "ATT on log monthly grid electricity consumption",
    fontsize=12,
    x=0.04,
)

###############################################################################
# SAVE
###############################################################################

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.10,
    top=0.95,
)

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PDF,
    format="pdf",
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", OUTPUT_PNG)
print("Saved:", OUTPUT_PDF)