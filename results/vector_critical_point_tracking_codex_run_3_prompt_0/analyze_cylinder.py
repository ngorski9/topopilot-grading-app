"""Track piecewise-linear vector-field critical points in the cylinder data."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import ot
import vtk
from vtk.util.numpy_support import vtk_to_numpy

DATA = Path("cylinder")
FRAMES = [1, 2, 3]


def read_frame(index):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(DATA / f"cylinder{index}.vti"))
    reader.Update()
    image = reader.GetOutput()
    nx, ny, _ = image.GetDimensions()
    u = vtk_to_numpy(image.GetPointData().GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(image.GetPointData().GetArray("v")).reshape(ny, nx)
    return u, v


def critical_points(u, v):
    """Return zeros of the linearly interpolated field in two triangles/cell."""
    ny, nx = u.shape
    points = []
    kinds = []
    # Counter-clockwise triangle vertex offsets, with a consistent grid diagonal.
    triangles = [((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))]
    for j in range(ny - 1):
        for i in range(nx - 1):
            for tri in triangles:
                xy = np.array([(i + dx, j + dy) for dx, dy in tri], float)
                f = np.array([[u[j + dy, i + dx], v[j + dy, i + dx]] for dx, dy in tri])
                # F(x,y) = f0 + J @ ((x,y) - xy0)
                A = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
                B = np.column_stack((f[1] - f[0], f[2] - f[0]))
                if abs(np.linalg.det(A)) < 1e-12:
                    continue
                J = B @ np.linalg.inv(A)
                if abs(np.linalg.det(J)) < 1e-10:
                    continue
                local = np.linalg.solve(J, -f[0])
                bary = np.array([1 - local.sum(), local[0], local[1]])
                # Half-open ownership avoids recording a zero on shared edges twice.
                if np.all(bary >= -1e-8) and np.all(bary <= 1 + 1e-8):
                    p = xy[0] + local
                    eig = np.linalg.eigvals(J)
                    if np.linalg.det(J) < 0:
                        kind = "saddle"
                    elif np.iscomplex(eig).any():
                        kind = "focus"
                    else:
                        kind = "node"
                    if not any(np.linalg.norm(p - q) < 1e-5 for q in points):
                        points.append(p)
                        kinds.append(kind)
    return np.asarray(points), kinds


def partial_match(a, b):
    """Partial OT correspondence, retaining the 95% lowest-cost mass."""
    if not len(a) or not len(b):
        return []
    cost = ot.dist(a, b, metric="sqeuclidean")
    # A distance scale makes the partial matching local and rejects distant births/deaths.
    cost /= max(cost.max(), 1.0)
    wa = np.ones(len(a)) / len(a)
    wb = np.ones(len(b)) / len(b)
    plan = ot.partial.partial_wasserstein(wa, wb, cost, m=0.95)
    rows, cols = np.where(plan > 1e-12)
    # Per-source strongest transport edge.  Only accept genuinely local assignments.
    matches = []
    for r in np.unique(rows):
        c = np.argmax(plan[r])
        if plan[r, c] > 1e-12 and np.linalg.norm(a[r] - b[c]) <= 25:
            matches.append((int(r), int(c), float(plan[r, c])))
    return matches


fields = [read_frame(t) for t in FRAMES]
cps = [critical_points(*f) for f in fields]
matches01 = partial_match(cps[0][0], cps[1][0])
matches12 = partial_match(cps[1][0], cps[2][0])
next12 = {a: b for a, b, _ in matches12}
tracks = [(a, b, next12[b]) for a, b, _ in matches01 if b in next12]

palette = {"saddle": "#d62728", "node": "#1f77b4", "focus": "#2ca02c"}
fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharex=True, sharey=True, constrained_layout=True)
for col, (ax, (u, v), (pts, types), frame) in enumerate(zip(axes, fields, cps, FRAMES)):
    speed = np.hypot(u, v)
    ax.imshow(speed, origin="lower", cmap="Greys", alpha=.72, extent=(0, u.shape[1]-1, 0, u.shape[0]-1))
    step = 8
    y, x = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
    ax.quiver(x, y, u[::step, ::step], v[::step, ::step], color="#406080", alpha=.62,
              scale=35, width=.002, headwidth=3)
    for kind in palette:
        sel = [k == kind for k in types]
        if any(sel):
            q = pts[sel]
            ax.scatter(q[:, 0], q[:, 1], s=42, c=palette[kind], edgecolors="white", linewidths=.6,
                       label=kind if col == 0 else None, zorder=5)
    ax.set_title(f"Time step {frame} ({len(pts)} critical points)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(loc="upper right", title="PL critical point")

# Overlay the complete three-step trajectories in every view, using time-colored vertices.
for ax in axes:
    for a, b, c in tracks:
        path = np.array([cps[0][0][a], cps[1][0][b], cps[2][0][c]])
        ax.plot(path[:, 0], path[:, 1], color="#ffbf00", lw=1.35, alpha=.9, zorder=4)
        ax.scatter(path[:, 0], path[:, 1], c=["#7b2cbf", "#ff7f0e", "#17becf"], s=18, zorder=6)
fig.suptitle(f"Cylinder vector field and partial-OT critical-point tracks ({len(tracks)} complete tracks)", fontsize=15)
fig.savefig("cylinder_critical_point_tracks.png", dpi=180, bbox_inches="tight")

result = {
    "frames": FRAMES,
    "critical_point_counts": [len(p[0]) for p in cps],
    "matches_1_to_2": len(matches01), "matches_2_to_3": len(matches12),
    "complete_three_step_tracks": len(tracks),
    "tracks": [{"t1": a, "t2": b, "t3": c,
                "positions": [cps[0][0][a].tolist(), cps[1][0][b].tolist(), cps[2][0][c].tolist()]}
               for a, b, c in tracks],
}
Path("cylinder_critical_point_tracks.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "tracks"}, indent=2))
