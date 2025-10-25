import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =====================
# Config
# =====================
arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485

# Adjustable parameters
tile_size = 0.7   # 🔧 Larger = bigger heatmap tiles (lower resolution)
arrow_size = 50   # 🔧 Larger = longer arrows

# =====================
# Load Data
# =====================
df = pd.read_csv("game_logs_bt_vs_fsm_2.csv")

# Filter: only Actor 0 (left-side bot)
df = df[df["Actor"].astype(str) == "0"]

# Drop invalid entries
df = df.dropna(subset=["BotPosX", "BotPosY", "BotRot"])

x = df["BotPosX"].values
y = df["BotPosY"].values
rot = df["BotRot"].values

# =====================
# Compute direction vectors
# =====================
rot_rad = np.radians(-rot)
u = np.sin(rot_rad)
v = np.cos(rot_rad)

# =====================
# Create plot
# =====================
fig, ax = plt.subplots(figsize=(8, 8))

# Draw arena boundary
circle = plt.Circle(arena_center, arena_radius, color="gray", fill=True, linestyle="-",edgecolor="black", linewidth=2,zorder=0)
ax.add_artist(circle)

# =====================
# Heatmap with adjustable tile size
# =====================
xrange = np.arange(x.min(), x.max() + tile_size, tile_size)
yrange = np.arange(y.min(), y.max() + tile_size, tile_size)
heatmap, xedges, yedges = np.histogram2d(x, y, bins=[xrange, yrange])

# fig.patch.set_facecolor("white")
# ax.set_facecolor("white")

ax.imshow(
    heatmap.T,
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap="Reds",
    alpha=0.9,
    aspect="equal",
    zorder=1
)

# =====================
# Direction field (quiver)
# =====================
# Downsample for visual clarity
step = max(1, len(df) // 200)  # show up to ~200 arrows

ax.quiver(
    x[::step],
    y[::step],
    u[::step],
    v[::step],
    color="black",
    alpha=0.5,
    scale=arrow_size,
    width=0.003,
    headwidth=3,
)

# =====================
# Labels & Arena Bounds
# =====================
ax.set_title("Sumobot Arena Heatmap + Direction Field")
ax.set_xlabel("BotPosX")
ax.set_ylabel("BotPosY")
ax.set_aspect("equal", adjustable="box")

ax.set_xlim(arena_center[0] - arena_radius - 1, arena_center[0] + arena_radius + 1)
ax.set_ylim(arena_center[1] - arena_radius - 1, arena_center[1] + arena_radius + 1)

plt.tight_layout()
plt.show()
