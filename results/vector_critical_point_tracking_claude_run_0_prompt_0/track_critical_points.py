"""
Track critical points of a time-varying, piecewise-linear 2D vector field
(TTK 'cylinder' dataset: arrays 'u','v' on a regular grid, one .vti per
timestep) across 3 consecutive timesteps using partial optimal transport.

Pipeline
--------
1. Load u,v for each requested timestep from the .vti (vtkImageData) files.
2. Triangulate each grid cell (2 triangles per quad) and, for every
   triangle, solve for the zero of the piecewise-linear (barycentric)
   interpolant of the vector field. A triangle contains a critical point
   iff the barycentric coordinates of the vector-value zero all lie in
   [0,1]. This is the standard combinatorial vector-field critical point
   extraction technique for PL fields.
3. Classify each critical point via the sign of the determinant / trace of
   the local Jacobian (estimated from the triangle's vector values) into
   sink, source or saddle.
4. Track critical points frame-to-frame with *partial* optimal transport
   (ot.partial.partial_wasserstein), which -- unlike full OT -- allows
   unequal masses on the two sides, so points that appear or disappear
   between frames are simply left unmatched instead of forcing a bad
   long-distance match.
5. Plot the vector field (as a streamplot) for each timestep with the
   critical points overlaid (colored by type) and the OT-tracked
   trajectories drawn as connecting lines across timesteps.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = "/workspace/cylinder"
TIMESTEPS = [1, 2, 3]          # 3 consecutive timesteps to track
DIST_CUTOFF = 15.0             # max matching distance (grid units); beyond -> unmatched
UNMATCHED_COST = DIST_CUTOFF   # "dustbin" cost for partial OT


def load_field(idx):
    fname = f"{DATA_DIR}/cylinder{idx}.vti"
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()
    img = reader.GetOutput()
    dims = img.GetDimensions()          # (nx, ny, 1)
    nx, ny = dims[0], dims[1]
    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(pd.GetArray("v")).reshape(ny, nx)
    return u, v, nx, ny


def find_critical_points(u, v):
    """Locate zeros of the PL-interpolated (u,v) field, cell by cell."""
    ny, nx = u.shape
    pts = []
    types = []

    def bary_zero(p0, p1, p2, f0, f1, f2):
        # Solve l0*f0 + l1*f1 + l2*f2 = 0, l0+l1+l2 = 1 for barycentric l.
        A = np.array([[f0[0], f1[0], f2[0]],
                      [f0[1], f1[1], f2[1]],
                      [1.0, 1.0, 1.0]])
        b = np.array([0.0, 0.0, 1.0])
        try:
            l = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        if np.all(l >= -1e-9) and np.all(l <= 1 + 1e-9):
            xy = l[0] * np.array(p0) + l[1] * np.array(p1) + l[2] * np.array(p2)
            return xy, l
        return None

    for j in range(ny - 1):
        for i in range(nx - 1):
            # grid coordinates (x=i, y=j)
            P00, P10, P01, P11 = (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            F00 = (u[j, i], v[j, i])
            F10 = (u[j, i + 1], v[j, i + 1])
            F01 = (u[j + 1, i], v[j + 1, i])
            F11 = (u[j + 1, i + 1], v[j + 1, i + 1])

            for (pa, pb, pc, fa, fb, fc) in (
                (P00, P10, P11, F00, F10, F11),
                (P00, P11, P01, F00, F11, F01),
            ):
                res = bary_zero(pa, pb, pc, fa, fb, fc)
                if res is None:
                    continue
                xy, _ = res
                # local Jacobian estimate from the triangle (constant per triangle)
                Mx = np.array([[pb[0] - pa[0], pc[0] - pa[0]],
                                [pb[1] - pa[1], pc[1] - pa[1]]])
                Du = np.array([fb[0] - fa[0], fc[0] - fa[0]])
                Dv = np.array([fb[1] - fa[1], fc[1] - fa[1]])
                try:
                    Minv = np.linalg.inv(Mx)
                except np.linalg.LinAlgError:
                    continue
                grad_u = Minv.T @ Du
                grad_v = Minv.T @ Dv
                J = np.array([grad_u, grad_v])
                det = np.linalg.det(J)
                trace = np.trace(J)
                if det < 0:
                    ctype = "saddle"
                elif trace < 0:
                    ctype = "sink"
                else:
                    ctype = "source"
                pts.append(xy)
                types.append(ctype)

    if not pts:
        return np.zeros((0, 2)), []

    pts = np.array(pts)
    # de-duplicate points shared between adjacent triangles (within 1e-6)
    keep = []
    seen = []
    for k, p in enumerate(pts):
        dup = False
        for q in seen:
            if np.linalg.norm(p - q) < 1e-6:
                dup = True
                break
        if not dup:
            seen.append(p)
            keep.append(k)
    return pts[keep], [types[k] for k in keep]


def partial_ot_match(P0, P1):
    """Match two point sets with partial optimal transport, allowing
    points to remain unmatched (birth/death of critical points)."""
    n0, n1 = len(P0), len(P1)
    if n0 == 0 or n1 == 0:
        return []

    C = ot.dist(P0, P1)  # squared euclidean by default -> use euclidean
    C = np.sqrt(C)
    # append dustbin row/col so partial_wasserstein can leave points unmatched
    a = np.ones(n0) / max(n0, n1)
    b = np.ones(n1) / max(n0, n1)
    m = min(a.sum(), b.sum())
    # fraction of mass to transport: only match points within DIST_CUTOFF
    # scale m down slightly to encourage leaving far points unmatched
    frac = m * (1.0 - 0.0)
    try:
        G = ot.partial.partial_wasserstein(a, b, C, m=frac)
    except Exception:
        return []

    matches = []
    for i in range(n0):
        j = np.argmax(G[i])
        if G[i, j] > 1e-12 and C[i, j] < DIST_CUTOFF:
            matches.append((i, j, C[i, j]))
    return matches


def main():
    frames = []
    for t in TIMESTEPS:
        u, v, nx, ny = load_field(t)
        pts, types = find_critical_points(u, v)
        frames.append(dict(t=t, u=u, v=v, nx=nx, ny=ny, pts=pts, types=types))
        print(f"timestep {t}: {len(pts)} critical points "
              f"({types.count('sink')} sinks, {types.count('source')} sources, "
              f"{types.count('saddle')} saddles)")

    # match consecutive frames with partial OT
    all_matches = []
    for k in range(len(frames) - 1):
        P0 = frames[k]["pts"]
        P1 = frames[k + 1]["pts"]
        matches = partial_ot_match(P0, P1) if len(P0) and len(P1) else []
        all_matches.append(matches)
        print(f"frame {frames[k]['t']} -> {frames[k+1]['t']}: {len(matches)} tracked matches")

    # build simple track ids: chain matches across the 3 frames
    n_frames = len(frames)
    track_id_per_frame = [np.full(len(frames[k]["pts"]), -1, dtype=int) for k in range(n_frames)]
    next_id = 0
    for i in range(len(frames[0]["pts"])):
        track_id_per_frame[0][i] = next_id
        next_id += 1

    for k in range(n_frames - 1):
        for (i, j, _) in all_matches[k]:
            track_id_per_frame[k + 1][j] = track_id_per_frame[k][i]
    # any still-unassigned point in frame>0 starts a new track (birth)
    for k in range(1, n_frames):
        for j in range(len(frames[k]["pts"])):
            if track_id_per_frame[k][j] == -1:
                track_id_per_frame[k][j] = next_id
                next_id += 1

    color_map = {"sink": "tab:blue", "source": "tab:red", "saddle": "tab:green"}

    fig, axes = plt.subplots(1, n_frames, figsize=(6 * n_frames, 6), sharey=True)
    if n_frames == 1:
        axes = [axes]

    ny, nx = frames[0]["ny"], frames[0]["nx"]
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    rng = np.random.default_rng(0)
    track_colors = {}

    for k, ax in enumerate(axes):
        fr = frames[k]
        ax.streamplot(X, Y, fr["u"], fr["v"], color="lightgray", density=1.2, linewidth=0.6, arrowsize=0.8)
        pts, types = fr["pts"], fr["types"]
        for ctype, c in color_map.items():
            mask = [t == ctype for t in types]
            if any(mask):
                ax.scatter(pts[mask, 0], pts[mask, 1], c=c, s=40, edgecolor="k",
                           zorder=5, label=ctype if k == 0 else None)
        for i, tid in enumerate(track_id_per_frame[k]):
            if tid not in track_colors:
                track_colors[tid] = rng.random(3)
            ax.annotate(str(tid), (pts[i, 0], pts[i, 1]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"timestep {fr['t']}")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("equal")

    # draw OT-tracked trajectories across frames on top of all subplots via
    # a shared figure-coordinate overlay is complex with separate axes, so
    # instead draw connecting arrows between adjacent subplots' matched pts
    # by plotting on a transform that spans figure coordinates.
    for k in range(n_frames - 1):
        axA, axB = axes[k], axes[k + 1]
        for (i, j, dist) in all_matches[k]:
            pA = frames[k]["pts"][i]
            pB = frames[k + 1]["pts"][j]
            tid = track_id_per_frame[k + 1][j]
            con = plt.matplotlib.patches.ConnectionPatch(
                xyA=pA, coordsA=axA.transData,
                xyB=pB, coordsB=axB.transData,
                color=track_colors.get(tid, "black"), lw=1.5, alpha=0.8, zorder=10)
            fig.add_artist(con)

    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                            markeredgecolor="k", markersize=8, label=t)
                    for t, c in color_map.items()]
    axes[0].legend(handles=legend_elems, loc="upper right")

    fig.suptitle("Cylinder vector field: critical points tracked over time via partial optimal transport")
    fig.tight_layout()
    out_path = "/workspace/cylinder_critical_points_tracking.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved figure to {out_path}")


if __name__ == "__main__":
    main()
