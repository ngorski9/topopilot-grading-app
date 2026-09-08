#!/opt/conda/bin/python3
"""Track PL critical points in the cylinder vector-field time series.

Writes a single PNG showing u/v streamlines and partial-OT tracks for the
first three time steps, plus CSV files with the detected points and matches.
"""
from pathlib import Path
import csv
import numpy as np
import ot
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cylinder"
STEPS = [1, 2, 3]


def read_field(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    extent = image.GetExtent()
    nx, ny = extent[1] - extent[0] + 1, extent[3] - extent[2] + 1
    u = vtk_to_numpy(image.GetPointData().GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(image.GetPointData().GetArray("v")).reshape(ny, nx)
    origin, spacing = image.GetOrigin(), image.GetSpacing()
    x = origin[0] + spacing[0] * np.arange(nx)
    y = origin[1] + spacing[1] * np.arange(ny)
    return x, y, u, v


def critical_points(x, y, u, v):
    """Find zeroes within each cell's two linear triangles.

    A vector field is affine on a triangle.  Its zero is therefore found by
    barycentric coordinates, retaining only strict interior points to prevent
    duplicate edge/vertex detections.
    """
    points = []
    ny, nx = u.shape
    # alternating diagonal prevents a grid-direction bias in the PL mesh.
    for j in range(ny - 1):
        for i in range(nx - 1):
            ids = (((0, 0, 1), (0, 1, 1)), ((0, 1, 1), (0, 0, 1))) if (i + j) % 2 == 0 else (((0, 0, 1), (0, 1, 0)), ((0, 1, 1), (1, 0, 1)))
            for di, dj in ids:
                vec = np.array([[u[j + dj[k], i + di[k]], v[j + dj[k], i + di[k]]] for k in range(3)])
                # v0 + a*(v1-v0) + b*(v2-v0) = 0
                try:
                    ab = np.linalg.solve(np.column_stack((vec[1] - vec[0], vec[2] - vec[0])), -vec[0])
                except np.linalg.LinAlgError:
                    continue
                bary = np.array([1.0 - ab.sum(), ab[0], ab[1]])
                if np.all(bary > 1e-8):
                    px = np.dot(bary, x[i + np.array(di)])
                    py = np.dot(bary, y[j + np.array(dj)])
                    # Jacobian of this affine triangle field; classify by det.
                    dxy = np.array([[x[i + di[1]] - x[i + di[0]], x[i + di[2]] - x[i + di[0]]],
                                    [y[j + dj[1]] - y[j + dj[0]], y[j + dj[2]] - y[j + dj[0]]]])
                    jac = np.column_stack((vec[1] - vec[0], vec[2] - vec[0])) @ np.linalg.inv(dxy)
                    points.append((px, py, float(np.linalg.det(jac))))
    return np.asarray(points)


def partial_matches(a, b, mass=0.85):
    """Partial OT: transport only the closest 85% of uniform point mass."""
    if len(a) == 0 or len(b) == 0:
        return []
    wa, wb = np.ones(len(a)) / len(a), np.ones(len(b)) / len(b)
    cost = ot.dist(a[:, :2], b[:, :2], metric="sqeuclidean")
    plan = ot.partial.partial_wasserstein(wa, wb, cost, m=mass)
    rows, cols = np.where(plan > 1e-10)
    # Each source/target has at most one dominant match for this sparse LP plan.
    best = {}
    for r, c in zip(rows, cols):
        if r not in best or plan[r, c] > plan[r, best[r]]:
            best[r] = c
    return [(r, c, float(plan[r, c]), float(cost[r, c])) for r, c in best.items()]


fields, cps = [], []
for step in STEPS:
    field = read_field(DATA / f"cylinder{step}.vti")
    fields.append(field)
    cps.append(critical_points(*field))

matches = [partial_matches(cps[k], cps[k + 1]) for k in range(len(STEPS) - 1)]

with open(ROOT / "cylinder_critical_points.csv", "w", newline="") as f:
    out = csv.writer(f); out.writerow(["time_step", "id", "x", "y", "jacobian_determinant", "type"])
    for step, pts in zip(STEPS, cps):
        for idx, (px, py, det) in enumerate(pts):
            out.writerow([step, idx, px, py, det, "saddle" if det < 0 else "node_or_focus"])
with open(ROOT / "cylinder_partial_ot_matches.csv", "w", newline="") as f:
    out = csv.writer(f); out.writerow(["from_step", "from_id", "to_step", "to_id", "transport_mass", "squared_distance"])
    for k, pairs in enumerate(matches):
        for r, c, w, d in pairs:
            out.writerow([STEPS[k], r, STEPS[k + 1], c, w, d])

fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
for k, (ax, step, field, pts) in enumerate(zip(axes, STEPS, fields, cps)):
    x, y, u, v = field
    speed = np.hypot(u, v)
    ax.imshow(speed, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis", alpha=.88)
    ss = 5
    ax.streamplot(x[::ss], y[::ss], u[::ss, ::ss], v[::ss, ::ss], color="white", density=1.25, linewidth=.45, arrowsize=.65)
    saddle = pts[:, 2] < 0 if len(pts) else []
    if len(pts):
        ax.scatter(pts[saddle, 0], pts[saddle, 1], marker="X", s=72, c="#ff3b30", edgecolors="black", linewidths=.55, label="saddle")
        ax.scatter(pts[~saddle, 0], pts[~saddle, 1], marker="o", s=56, c="#00e5ff", edgecolors="black", linewidths=.55, label="node/focus")
    # Show incoming and outgoing partial-OT links locally, avoiding visual
    # ambiguity caused by plotting across separate time-step panels.
    for links, endpoint, side in ((matches[k - 1] if k else [], 1, "in"), (matches[k] if k < 2 else [], 0, "out")):
        for r, c, mass, _ in links:
            a, b = (cps[k - 1][r], pts[c]) if side == "in" else (pts[r], cps[k + 1][c])
            ax.annotate("", xy=(a[0], a[1]), xytext=(b[0], b[1]), arrowprops=dict(arrowstyle="->", color="#ffcc00", lw=.75, alpha=.8))
    ax.set_title(f"Cylinder field — time step {step} ({len(pts)} critical points)")
    ax.set_xlabel("x"); ax.set_aspect("equal")
axes[0].set_ylabel("y")
axes[0].legend(loc="upper right", framealpha=.8)
fig.suptitle("Piecewise-linear critical points and partial optimal-transport tracking (85% mass)", fontsize=15)
fig.savefig(ROOT / "cylinder_critical_points_partial_ot.png", dpi=190)
print("critical point counts:", [len(p) for p in cps])
print("partial-OT matches:", [len(m) for m in matches])
