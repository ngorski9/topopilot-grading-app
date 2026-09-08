"""
Track critical points of a time-varying 2D piecewise-linear vector field
(components 'u','v' stored on a structured grid, .vti files) across 3
timesteps, using partial optimal transport (POT) to match critical points
between consecutive frames.

Pipeline:
  1. Load 3 timesteps of the cylinder dataset (u, v point-data arrays).
  2. Extract critical points of the PL vector field: for every triangle of
     the (implicit) triangulated grid, solve for the barycentric location
     where the linearly-interpolated vector (u,v) vanishes. Classify each
     critical point (sink / source / saddle / center) from the local
     Jacobian.
  3. Match critical points between consecutive timesteps with *partial*
     optimal transport (ot.partial.partial_wasserstein): this allows some
     critical points to be left unmatched (birth/death events) instead of
     forcing a full assignment, which is what plain OT / the Hungarian
     algorithm would do.
  4. Chain the pairwise matches into trajectories across the 3 timesteps.
  5. Visualize: vector field (quiver + magnitude background) of the last
     timestep, all critical points (colored by type, faded by time), and
     the tracked trajectories drawn as connecting lines/arrows.
"""
import glob
import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = "./cylinder"
N_STEPS = 3
OUT_PNG = "./cylinder_critical_point_tracking.png"


def load_timestep(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    img = reader.GetOutput()
    nx, ny, nz = img.GetDimensions()
    assert nz == 1, "expected a 2D slice"
    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(pd.GetArray("v")).reshape(ny, nx)
    ox, oy, oz = img.GetOrigin()
    sx, sy, sz = img.GetSpacing()
    xs = ox + sx * np.arange(nx)
    ys = oy + sy * np.arange(ny)
    return xs, ys, u, v


def extract_critical_points(xs, ys, u, v):
    """Find zeros of the PL-interpolated vector field on the grid's
    triangulation (each quad cell split into 2 triangles), classify them."""
    ny, nx = u.shape
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)

    def gather(i0, j0, i1, j1, i2, j2):
        # points p0,p1,p2 and vector values w0,w1,w2 for a batch of triangles
        p0 = np.stack([X[j0, i0], Y[j0, i0]], axis=-1)
        p1 = np.stack([X[j1, i1], Y[j1, i1]], axis=-1)
        p2 = np.stack([X[j2, i2], Y[j2, i2]], axis=-1)
        w0 = np.stack([u[j0, i0], v[j0, i0]], axis=-1)
        w1 = np.stack([u[j1, i1], v[j1, i1]], axis=-1)
        w2 = np.stack([u[j2, i2], v[j2, i2]], axis=-1)
        return p0, p1, p2, w0, w1, w2

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    ii = ii.ravel()
    jj = jj.ravel()

    # 4 corners of each quad
    i00, j00 = ii, jj
    i10, j10 = ii + 1, jj
    i01, j01 = ii, jj + 1
    i11, j11 = ii + 1, jj + 1

    triangles = [
        (i00, j00, i10, j10, i11, j11),
        (i00, j00, i11, j11, i01, j01),
    ]

    crit_pts = []
    crit_types = []

    for (a_i, a_j, b_i, b_j, c_i, c_j) in triangles:
        p0, p1, p2, w0, w1, w2 = gather(a_i, a_j, b_i, b_j, c_i, c_j)

        # solve M [s,t]^T = -w0 where M columns are (w1-w0),(w2-w0)
        m00 = w1[:, 0] - w0[:, 0]
        m01 = w2[:, 0] - w0[:, 0]
        m10 = w1[:, 1] - w0[:, 1]
        m11 = w2[:, 1] - w0[:, 1]
        det = m00 * m11 - m01 * m10
        valid = np.abs(det) > 1e-14

        rhs0 = -w0[:, 0]
        rhs1 = -w0[:, 1]
        s = np.zeros_like(det)
        t = np.zeros_like(det)
        s[valid] = (rhs0[valid] * m11[valid] - m01[valid] * rhs1[valid]) / det[valid]
        t[valid] = (m00[valid] * rhs1[valid] - rhs0[valid] * m10[valid]) / det[valid]

        eps = 1e-9
        inside = valid & (s >= -eps) & (t >= -eps) & (s + t <= 1 + eps)
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            continue

        pos = p0[idx] + (p1[idx] - p0[idx]) * s[idx, None] + (p2[idx] - p0[idx]) * t[idx, None]

        # local (constant) Jacobian of the linear field on this triangle,
        # from edge vectors in physical space vs. value differences
        e1 = p1[idx] - p0[idx]
        e2 = p2[idx] - p0[idx]
        du = np.stack([w1[idx, 0] - w0[idx, 0], w2[idx, 0] - w0[idx, 0]], axis=-1)
        dv = np.stack([w1[idx, 1] - w0[idx, 1], w2[idx, 1] - w0[idx, 1]], axis=-1)
        E = np.stack([e1, e2], axis=-1)  # (n,2,2), columns e1,e2
        Einv = np.linalg.inv(E)
        grad_u = np.einsum('nij,ni->nj', Einv, du)  # [du/dx, du/dy]
        grad_v = np.einsum('nij,ni->nj', Einv, dv)

        trace = grad_u[:, 0] + grad_v[:, 1]
        deter = grad_u[:, 0] * grad_v[:, 1] - grad_u[:, 1] * grad_v[:, 0]
        disc = trace ** 2 - 4 * deter

        ctype = np.empty(idx.size, dtype=object)
        ctype[deter < 0] = "saddle"
        node_mask = (deter >= 0) & (disc >= 0)
        focus_mask = (deter >= 0) & (disc < 0)
        ctype[node_mask & (trace < 0)] = "sink"
        ctype[node_mask & (trace >= 0)] = "source"
        ctype[focus_mask & (trace < 0)] = "attracting_focus"
        ctype[focus_mask & (trace >= 0)] = "repelling_focus"

        crit_pts.append(pos)
        crit_types.append(ctype)

    if not crit_pts:
        return np.zeros((0, 2)), np.array([], dtype=object)

    pts = np.concatenate(crit_pts, axis=0)
    types = np.concatenate(crit_types, axis=0)

    # de-duplicate points that were found on both triangles sharing an edge
    # or that coincide with grid vertices (shared corner of 4 quads)
    rounded = np.round(pts, 6)
    _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return pts[uniq_idx], types[uniq_idx]


TYPE_COLOR = {
    "sink": "tab:blue",
    "source": "tab:red",
    "saddle": "tab:green",
    "attracting_focus": "tab:cyan",
    "repelling_focus": "tab:orange",
}


def match_partial_ot(pts_a, pts_b):
    """Match critical points between two frames with partial optimal
    transport: mass to transport = min(#a, #b), leaving extras unmatched
    (births/deaths) rather than forcing a full assignment."""
    na, nb = len(pts_a), len(pts_b)
    if na == 0 or nb == 0:
        return []

    cost = ot.dist(pts_a, pts_b, metric="sqeuclidean")
    a = np.ones(na) / na
    b = np.ones(nb) / nb
    m = min(na, nb) / max(na, nb) * min(np.sum(a), np.sum(b))
    # transport at most m units of mass (partial OT)
    gamma = ot.partial.partial_wasserstein(a, b, cost, m=m)

    # reject matches whose spatial jump is implausibly large
    dists = np.sqrt(cost)
    max_jump = 0.15 * max(pts_a[:, 0].max() - pts_a[:, 0].min(),
                           pts_a[:, 1].max() - pts_a[:, 1].min() + 1e-9)

    matches = []
    thresh = gamma.max() * 1e-6 if gamma.max() > 0 else 0
    for i in range(na):
        j = np.argmax(gamma[i])
        if gamma[i, j] > thresh and dists[i, j] < max_jump:
            matches.append((i, j))
    return matches


def main():
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "cylinder*.vti")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("cylinder", "")),
    )[:N_STEPS]
    print("Using timesteps:", files)

    frames = [load_timestep(f) for f in files]

    all_pts, all_types = [], []
    for xs, ys, u, v in frames:
        pts, types = extract_critical_points(xs, ys, u, v)
        all_pts.append(pts)
        all_types.append(types)
        print(f"  found {len(pts)} critical points "
              f"({', '.join(f'{t}:{int((types==t).sum())}' for t in TYPE_COLOR)})")

    # match consecutive frames with partial optimal transport
    pairwise_matches = []
    for k in range(N_STEPS - 1):
        m = match_partial_ot(all_pts[k], all_pts[k + 1])
        pairwise_matches.append(m)
        print(f"  matched {len(m)} critical points between step {k} and {k + 1} (partial OT)")

    # chain matches step0->step1->step2 into trajectories
    trajectories = []  # list of lists of (step, idx)
    map01 = dict(pairwise_matches[0]) if N_STEPS > 1 else {}
    map12 = dict(pairwise_matches[1]) if N_STEPS > 2 else {}
    matched_1_from_0 = set(map01.values())
    matched_0 = set(map01.keys())

    for i0 in range(len(all_pts[0])):
        traj = [(0, i0)]
        if i0 in map01:
            i1 = map01[i0]
            traj.append((1, i1))
            if N_STEPS > 2 and i1 in map12:
                traj.append((2, map12[i1]))
        trajectories.append(traj)

    # unmatched points born at step 1 (not matched from step 0)
    for i1 in range(len(all_pts[1]) if N_STEPS > 1 else 0):
        if i1 not in matched_1_from_0:
            traj = [(1, i1)]
            if N_STEPS > 2 and i1 in map12:
                traj.append((2, map12[i1]))
            trajectories.append(traj)

    # unmatched points born fresh at step 2
    matched_2 = set(map12.values())
    for i2 in range(len(all_pts[2]) if N_STEPS > 2 else 0):
        if i2 not in matched_2:
            trajectories.append([(2, i2)])

    n_tracked = sum(1 for t in trajectories if len(t) == N_STEPS)
    print(f"  {n_tracked} critical points tracked across all {N_STEPS} timesteps")

    visualize(files, frames, all_pts, all_types, trajectories)


def visualize(files, frames, all_pts, all_types, trajectories):
    xs, ys, u_last, v_last = frames[-1]
    X, Y = np.meshgrid(xs, ys)
    mag = np.sqrt(u_last ** 2 + v_last ** 2)

    fig, ax = plt.subplots(figsize=(6, 14))
    ax.pcolormesh(X, Y, mag, shading="auto", cmap="Greys", alpha=0.6)

    step = 6
    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        u_last[::step, ::step], v_last[::step, ::step],
        color="steelblue", scale=40, width=0.0025, alpha=0.8,
    )

    alphas = np.linspace(0.35, 1.0, len(frames))
    for k, (pts, types) in enumerate(zip(all_pts, all_types)):
        for t, color in TYPE_COLOR.items():
            sel = types == t
            if sel.any():
                ax.scatter(
                    pts[sel, 0], pts[sel, 1],
                    color=color, alpha=alphas[k],
                    s=60 if k == len(frames) - 1 else 30,
                    edgecolors="black", linewidths=0.4,
                    zorder=3,
                )

    for traj in trajectories:
        if len(traj) < 2:
            continue
        coords = np.array([all_pts[s][i] for s, i in traj])
        ax.plot(coords[:, 0], coords[:, 1], "-", color="black", linewidth=1.2,
                 alpha=0.8, zorder=2)
        ax.annotate(
            "", xy=coords[-1], xytext=coords[-2],
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, alpha=0.8),
            zorder=2,
        )

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
               label=t, markersize=8)
        for t, c in TYPE_COLOR.items()
    ]
    legend_elems.append(Line2D([0], [0], color="black", lw=1.5, label="tracked trajectory"))
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(
        "Critical points of the time-varying cylinder vector field\n"
        f"tracked over {len(files)} timesteps via partial optimal transport"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170)
    print(f"Saved visualization to {OUT_PNG}")


if __name__ == "__main__":
    main()
