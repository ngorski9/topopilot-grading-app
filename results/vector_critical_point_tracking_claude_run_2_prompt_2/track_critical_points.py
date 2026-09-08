"""
Track critical points of the time-varying 2D piecewise-linear vector field
(u, v) from the ./cylinder dataset across 3 time steps, using partial
optimal transport (POT) for the matching, and visualize the result together
with the original vector field.

Run with: pvpython track_critical_points.py
"""
import numpy as np
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import vtk
from vtk.util.numpy_support import vtk_to_numpy

# ----------------------------------------------------------------------
# 0. Choose 3 time steps spread across the series so the flow visibly
#    evolves (vortex shedding) between them.
# ----------------------------------------------------------------------
TIMESTEPS = [1, 11, 21]
FILES = [f"/workspace/cylinder/cylinder{t}.vti" for t in TIMESTEPS]


def load_uv(path):
    """Read a .vti file and return u, v as 2D numpy arrays (ny, nx) plus grid spacing/origin."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    img = reader.GetOutput()

    dims = img.GetDimensions()  # (nx, ny, 1)
    nx, ny = dims[0], dims[1]
    origin = img.GetOrigin()
    spacing = img.GetSpacing()

    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(pd.GetArray("v")).reshape(ny, nx)

    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    return u, v, xs, ys


# ----------------------------------------------------------------------
# 1. Extract critical points of the PL vector field on the grid.
#
# Each grid quad is split into 2 triangles. On a triangle the field is
# affine, so it maps the (physical) triangle onto a triangle in
# (u, v)-value space. There is a critical point inside the domain
# triangle iff the origin (0, 0) lies inside the value-space triangle;
# its position is recovered from the barycentric coordinates of the
# origin, and its type from the local Jacobian (finite differences).
# ----------------------------------------------------------------------
def barycentric_of_origin(p0, p1, p2):
    """Barycentric coords (l0, l1, l2) of the origin wrt triangle p0,p1,p2 in value space."""
    T = np.array([[p0[0] - p2[0], p1[0] - p2[0]],
                  [p0[1] - p2[1], p1[1] - p2[1]]])
    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    if abs(det) < 1e-14:
        return None
    rhs = np.array([-p2[0], -p2[1]])
    l0 = (T[1, 1] * rhs[0] - T[0, 1] * rhs[1]) / det
    l1 = (-T[1, 0] * rhs[0] + T[0, 0] * rhs[1]) / det
    l2 = 1.0 - l0 - l1
    return l0, l1, l2, det


def classify(jac):
    """Classify a critical point from its 2x2 Jacobian [[dudx,dudy],[dvdx,dvdy]]."""
    tr = jac[0, 0] + jac[1, 1]
    det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
    disc = tr * tr - 4 * det
    if det < 0:
        return "saddle"
    if disc >= 0:
        return "sink" if tr < 0 else "source"
    return "attracting focus" if tr < 0 else "repelling focus"


def extract_critical_points(u, v, xs, ys):
    ny, nx = u.shape
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # cell-centered finite-difference Jacobian, computed once on the grid
    dudx, dudy = np.gradient(u, dx, dy, axis=(1, 0))
    dvdx, dvdy = np.gradient(v, dx, dy, axis=(1, 0))

    # the cylinder obstacle is masked with an exact (0,0) vector; any triangle
    # touching that masked region is degenerate and must be excluded, or it
    # spuriously reports a "critical point" along the whole obstacle boundary
    blanked = (u == 0.0) & (v == 0.0)

    points = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if blanked[j, i] or blanked[j, i + 1] or blanked[j + 1, i] or blanked[j + 1, i + 1]:
                continue
            # corners of the quad cell (i,j)-(i+1,j+1)
            idx = {
                "00": (i, j), "10": (i + 1, j), "11": (i + 1, j + 1), "01": (i, j + 1),
            }
            P = {k: (xs[ci], ys[cj]) for k, (ci, cj) in idx.items()}
            V = {k: (u[cj, ci], v[cj, ci]) for k, (ci, cj) in idx.items()}
            # split quad into 2 triangles (00,10,11) and (00,11,01)
            for tri in [("00", "10", "11"), ("00", "11", "01")]:
                bc = barycentric_of_origin(V[tri[0]], V[tri[1]], V[tri[2]])
                if bc is None:
                    continue
                l0, l1, l2, det = bc
                if l0 >= 0 and l1 >= 0 and l2 >= 0:
                    px = l0 * P[tri[0]][0] + l1 * P[tri[1]][0] + l2 * P[tri[2]][0]
                    py = l0 * P[tri[0]][1] + l1 * P[tri[1]][1] + l2 * P[tri[2]][1]
                    # local Jacobian: average the 3 corner FD-Jacobians, weighted by barycentric coords
                    jac = np.zeros((2, 2))
                    for k, w in zip(tri, (l0, l1, l2)):
                        ci, cj = idx[k]
                        jac += w * np.array([[dudx[cj, ci], dudy[cj, ci]],
                                             [dvdx[cj, ci], dvdy[cj, ci]]])
                    ctype = classify(jac)
                    index = -1 if ctype == "saddle" else 1
                    points.append({"x": px, "y": py, "type": ctype, "index": index})
    # de-duplicate points that are essentially coincident (can happen at shared triangle edges)
    dedup = []
    for p in points:
        if not any(abs(p["x"] - q["x"]) < 1e-6 and abs(p["y"] - q["y"]) < 1e-6 for q in dedup):
            dedup.append(p)
    return dedup


# ----------------------------------------------------------------------
# 2. Partial optimal transport matching between consecutive time steps.
# ----------------------------------------------------------------------
TYPE_PENALTY = 1e4  # forbids matching a saddle to a sink/source etc.


def cost_matrix(cps_a, cps_b):
    xa = np.array([[p["x"], p["y"]] for p in cps_a])
    xb = np.array([[p["x"], p["y"]] for p in cps_b])
    C = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=2)
    for i, pa in enumerate(cps_a):
        for j, pb in enumerate(cps_b):
            if pa["type"] != pb["type"]:
                C[i, j] += TYPE_PENALTY
    return C


def match_partial_ot(cps_a, cps_b):
    """Match critical points between two time steps with partial optimal transport.

    Uniform mass 1 is placed on each point; since the two point clouds can
    have a different number of points (critical points are born/destroyed),
    only min(len(a), len(b)) units of mass are transported -- the rest is
    legitimately left unmatched (births/deaths).
    """
    na, nb = len(cps_a), len(cps_b)
    if na == 0 or nb == 0:
        return []
    a = np.ones(na) / na
    b = np.ones(nb) / nb
    C = cost_matrix(cps_a, cps_b)
    m = min(na, nb) / max(na, nb) * min(np.sum(a), np.sum(b))
    # mass to transport: match every point of the smaller set (partial OT)
    m = min(na, nb) * (1.0 / max(na, nb))
    gamma = ot.partial.partial_wasserstein(a, b, C, m=m)

    matches = []
    used_a, used_b = set(), set()
    # greedily read off the transport plan: for each row take its heaviest column
    for i in range(na):
        row = gamma[i]
        if row.max() <= 1e-12:
            continue
        j = int(np.argmax(row))
        if C[i, j] >= TYPE_PENALTY:  # forbidden (type mismatch) match, ignore
            continue
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j))
    return matches


# ----------------------------------------------------------------------
# 3. Run the pipeline: load, extract, match, assign track ids.
# ----------------------------------------------------------------------
fields = [load_uv(f) for f in FILES]
cps_per_step = [extract_critical_points(u, v, xs, ys) for (u, v, xs, ys) in fields]

for t, cps in zip(TIMESTEPS, cps_per_step):
    print(f"time step {t}: {len(cps)} critical points "
          f"({sum(1 for c in cps if c['type']=='saddle')} saddle, "
          f"{sum(1 for c in cps if 'sink' in c['type'] or 'attracting' in c['type'])} sink-like, "
          f"{sum(1 for c in cps if 'source' in c['type'] or 'repelling' in c['type'])} source-like)")

# assign a persistent track id to every critical point, propagated by matching
next_track_id = 0
track_ids_per_step = []
for cps in cps_per_step:
    track_ids_per_step.append([None] * len(cps))

for i, p in enumerate(cps_per_step[0]):
    track_ids_per_step[0][i] = next_track_id
    next_track_id += 1

all_matches = []  # list of (step_index, i, j) pairs between step and step+1
for s in range(len(cps_per_step) - 1):
    matches = match_partial_ot(cps_per_step[s], cps_per_step[s + 1])
    all_matches.append(matches)
    for (i, j) in matches:
        track_ids_per_step[s + 1][j] = track_ids_per_step[s][i]

# points created at step s+1 with no incoming match get a fresh track id
for s in range(1, len(cps_per_step)):
    for j in range(len(cps_per_step[s])):
        if track_ids_per_step[s][j] is None:
            track_ids_per_step[s][j] = next_track_id
            next_track_id += 1

print(f"\ntotal distinct tracks over {len(TIMESTEPS)} time steps: {next_track_id}")
for s in range(len(all_matches)):
    print(f"  matched {len(all_matches[s])}/{min(len(cps_per_step[s]), len(cps_per_step[s+1]))} "
          f"possible pairs between t={TIMESTEPS[s]} and t={TIMESTEPS[s+1]}")

# ----------------------------------------------------------------------
# 4. Visualization: vector field (as a streamplot/quiver background) at
#    each time step, with the tracked critical points overlaid and
#    connected by their trajectories across time steps.
# ----------------------------------------------------------------------
type_color = {
    "saddle": "gold",
    "sink": "royalblue",
    "source": "crimson",
    "attracting focus": "mediumpurple",
    "repelling focus": "darkorange",
}

fig, axes = plt.subplots(1, len(TIMESTEPS), figsize=(6 * len(TIMESTEPS), 6), sharex=True, sharey=True)

# build per-track trajectories (list of (x,y) per step it exists at, None if absent)
track_traj = {tid: [None] * len(TIMESTEPS) for tid in range(next_track_id)}
for s, cps in enumerate(cps_per_step):
    for i, p in enumerate(cps):
        tid = track_ids_per_step[s][i]
        track_traj[tid][s] = (p["x"], p["y"])

for s, (ax, (u, v, xs, ys), cps) in enumerate(zip(axes, fields, cps_per_step)):
    mag = np.sqrt(u ** 2 + v ** 2)
    ax.imshow(mag, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
              cmap="Greys", alpha=0.6)
    step = 6
    ax.quiver(xs[::step], ys[::step], u[::step, ::step], v[::step, ::step],
              color="black", scale=200, width=0.002, alpha=0.7)

    for p, tid in zip(cps, track_ids_per_step[s]):
        ax.scatter(p["x"], p["y"], s=90, edgecolor="black", linewidth=0.8,
                   color=type_color.get(p["type"], "grey"), zorder=5)
        ax.annotate(str(tid), (p["x"], p["y"]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, zorder=6)

    # draw the portion of every trajectory that connects this step to the previous one
    if s > 0:
        segs, cols = [], []
        for tid, traj in track_traj.items():
            if traj[s - 1] is not None and traj[s] is not None:
                segs.append([traj[s - 1], traj[s]])
                cols.append("lime")
        if segs:
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=2, alpha=0.9, zorder=4))

    ax.set_title(f"t = {TIMESTEPS[s]}  ({len(cps)} critical points)")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                      label=t, markersize=9) for t, c in type_color.items()]
handles.append(plt.Line2D([0], [0], color="lime", lw=2, label="track (partial-OT match)"))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Critical points of the (u,v) vector field tracked across time steps "
             f"{TIMESTEPS} via partial optimal transport", y=1.02)
fig.tight_layout(rect=[0, 0.06, 1, 1])
out_path = "/workspace/cylinder_critical_point_tracking.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved visualization to {out_path}")
