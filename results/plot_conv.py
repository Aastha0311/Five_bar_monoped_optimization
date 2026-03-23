# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load data
# df = pd.read_csv("/home/stochlab/repo/optimal-design-legged-robots/results/planar/dist/20_all/11_zeros/shank mass map/no_landing/best_dist_20_079_020_2026-03-22_06-27-04_5.0.csv")

# # ---- ITERATION INDEX ----
# df = df.reset_index(drop=True)
# df["iteration"] = df.index

# # ---- COST ----
# cost = df["best_cost"].values

# # ---- BEST-SO-FAR (cumulative minimum) ----
# best_so_far = np.minimum.accumulate(cost)

# # ---- PLOT SETTINGS ----
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 12,
#     "figure.figsize": (12, 9),
#     "lines.linewidth": 1.5,
#     "lines.markersize": 4
# })

# fig, ax = plt.subplots()

# # ---- RAW COST (scatter + light line) ----
# ax.plot(
#     df["iteration"],
#     cost,
#     marker='o',
#     linestyle='-',
#     alpha=0.2,
#     label="Cost per iteration"
# )

# # ---- BEST EVER CURVE ----
# ax.plot(
#     df["iteration"],
#     best_so_far,
#     linestyle='-',
#     linewidth=2,
#     label="Best so far"
# )

# # ---- MARK GLOBAL BEST POINT ----
# best_idx = np.argmin(cost)
# ax.scatter(
#     best_idx,
#     cost[best_idx],
#     s=60,
#     zorder=5,
#     label=f"Best @ iter {best_idx}"
# )

# # ---- LABELS ----
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Cost")

# # ---- GRID + LEGEND ----
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend()

# # ---- FINAL ----
# plt.tight_layout()

# plt.savefig("cost_vs_iterations.pdf")
# plt.savefig("cost_vs_iterations.png", dpi=300)

# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("/home/stochlab/repo/optimal-design-legged-robots/results/planar/dist/20_all/11_zeros/shank mass map/no_landing/best_dist_20_079_020_2026-03-22_06-27-04_5.0.csv")

# Iterations
df = df.reset_index(drop=True)
df["iteration"] = df.index

cost = df["best_cost"].values

# Best-so-far (cumulative minimum)
best_so_far = np.minimum.accumulate(cost)

# ---- Plot settings ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.figsize": (12, 9),
})

fig, ax = plt.subplots()

# ---- Scatter only (NO line) ----
ax.scatter(
    df["iteration"],
    cost,
    alpha=0.6,
    s=20,
    label="Cost per iteration"
)

# ---- Best-so-far curve (keep line) ----
ax.plot(
    df["iteration"],
    best_so_far,
    linewidth=2,
    label="Best so far", 
    color='red'
)

# ---- Highlight global best ----
best_idx = np.argmin(cost)
# ax.scatter(
#     best_idx,
#     cost[best_idx],
#     s=60,
#     zorder=5,
#     label=f"Best @ iter {best_idx}"
# )

# Labels
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")

# Grid + legend
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("cost_vs_iterations.pdf")
plt.show()