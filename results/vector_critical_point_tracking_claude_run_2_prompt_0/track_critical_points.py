"""
Track critical points of a time-varying 2D vector field (TTK 'cylinder' dataset)
across 3 time steps using partial optimal transport, and visualize the tracks
together with the underlying vector field.

Pipeline
--------
1. Read u, v point-data arrays from consecutive .vti files (vtkXMLImageDataReader).
2. Find critical points of the piecewise-linear vector field: each grid cell is
   split into two triangles; on each triangle (u, v) is linear, so we solve for
   the barycentric coordinates at which the linear interpolant is exactly (0, 0).
   A valid critical point exists if all barycentric coordinates lie in [0, 1].
3. Classify each critical point (sink / source / saddle / center) from the sign
   pattern of the eigenvalues of the (constant, per-triangle) Jacobian of (u, v).
4. Track critical points frame-to-frame with *partial* optimal transport
   (ot.partial.partial_wasserstein), which allows points to appear/disappear
   between frames (mass need not be fully transported) -- appropriate since
   the critical point count changes over time (vortices are born/die).
5. Render the vector field (streamlines colored by speed) for each of the 3
   frames with the classified critical points overlaid, plus the transport
   links connecting matched critical points between consecutive frames.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = "/workspace/cylinder"
TIMESTEPS = [1, 2, 3]  # 3 consecutive time steps to track


def read_field(idx):
    fname = f"{DATA_DIR}/cylinder{idx}.vti"
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()
    img = reader.GetOutput()
    nx, ny, nz = img.GetDimensions()
    ox, oy, oz = img.GetOrigin()
    sx, sy, sz = img.GetSpacing()
    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(pd.GetArray("v")).reshape(ny, nx)
    xs = ox + sx * np.arange(nx)
    ys = oy + sy * np.arange(ny)
    return xs, ys, u, v


def triangle_critical_point(p0, p1, p2, f0, f1, f2, eps=1e-9):
    """Solve l0*f0 + l1*f1 + l2*f2 = 0, l0+l1+l2=1 for barycentric coords.
    Returns (point, jacobian) if the zero lies inside the triangle, else None."""
    M = np.array([[f0[0] - f2[0], f1[0] - f2[0]],
                  [f0[1] - f2[1], f1[1] - f2[1]]])
    rhs = -np.array(f2)
    det = np.linalg.det(M)
    if abs(det) < eps:
        return None
    l0, l1 = np.linalg.solve(M, rhs)
    l2 = 1.0 - l0 - l1
    if l0 < -1e-6 or l1 < -1e-6 or l2 < -1e-6:
        return None
    if l0 > 1 + 1e-6 or l1 > 1 + 1e-6 or l2 > 1 + 1e-6:
        return None
    pos = l0 * np.array(p0) + l1 * np.array(p1) + l2 * np.array(p2)

    # Constant Jacobian of the linear interpolant over this triangle:
    # solve for gradients of u and v via the triangle's edge geometry.
    A = np.array([[p1[0] - p0[0], p1[1] - p0[1]],
                  [p2[0] - p0[0], p2[1] - p0[1]]])
    du = np.array([f1[0] - f0[0], f2[0] - f0[0]])
    dv = np.array([f1[1] - f0[1], f2[1] - f0[1]])
    grad_u = np.linalg.solve(A, du)
    grad_v = np.linalg.solve(A, dv)
    J = np.array([grad_u, grad_v])
    return pos, J


def classify(J):
    eig = np.linalg.eigvals(J)
    if np.iscomplexobj(eig) and np.any(np.abs(eig.imag) > 1e-8):
        return "center/spiral"
    re = eig.real
    if np.all(re > 1e-8):
        return "source"
    if np.all(re < -1e-8):
        return "sink"
    return "saddle"


def find_critical_points(xs, ys, u, v):
    ny, nx = u.shape
    pts, types = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p00 = (xs[i], ys[j]); p10 = (xs[i + 1], ys[j])
            p01 = (xs[i], ys[j + 1]); p11 = (xs[i + 1], ys[j + 1])
            f00 = (u[j, i], v[j, i]); f10 = (u[j, i + 1], v[j, i + 1])
            f01 = (u[j + 1, i], v[j + 1, i]); f11 = (u[j + 1, i + 1], v[j + 1, i + 1])
            for (a, b, c, fa, fb, fc) in [
                (p00, p10, p11, f00, f10, f11),
                (p00, p11, p01, f00, f11, f01),
            ]:
                res = triangle_critical_point(a, b, c, fa, fb, fc)
                if res is not None:
                    pos, J = res
                    pts.append(pos)
                    types.append(classify(J))
    return np.array(pts) if pts else np.empty((0, 2)), types


def partial_ot_track(P0, P1, max_dist=25.0):
    """Match critical points between two frames with partial optimal transport,
    allowing unmatched (appearing/disappearing) points."""
    n0, n1 = len(P0), len(P1)
    if n0 == 0 or n1 == 0:
        return []
    C = ot.dist(P0, P1, metric="euclidean")
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    m = min(n0, n1) / max(n0, n1) * min(1.0, 1.0)  # fraction of mass to transport
    m = 0.9 * min(a.sum(), b.sum())
    G = ot.partial.partial_wasserstein(a, b, C, m=m)
    links = []
    for i in range(n0):
        j = np.argmax(G[i])
        if G[i, j] > 1e-12 and C[i, j] < max_dist:
            links.append((i, j))
    return links


COLORS = {"sink": "tab:blue", "source": "tab:red",
          "saddle": "tab:green", "center/spiral": "tab:purple"}

frames = []
for t in TIMESTEPS:
    xs, ys, u, v = read_field(t)
    pts, types = find_critical_points(xs, ys, u, v)
    frames.append(dict(t=t, xs=xs, ys=ys, u=u, v=v, pts=pts, types=types))
    print(f"t={t}: {len(pts)} critical points -> "
          + ", ".join(f"{k}:{types.count(k)}" for k in set(types)))

link_frames = []
for k in range(len(frames) - 1):
    P0, P1 = frames[k]["pts"], frames[k + 1]["pts"]
    links = partial_ot_track(P0, P1)
    link_frames.append(links)
    print(f"tracked {len(links)} critical points from t={frames[k]['t']} "
          f"to t={frames[k+1]['t']} (out of {len(P0)} / {len(P1)})")

# assign a persistent track id to every matched chain, propagated across frames
track_id_maps = [dict() for _ in frames]
next_id = 0
for i in range(len(frames[0]["pts"])):
    track_id_maps[0][i] = next_id
    next_id += 1
for k, links in enumerate(link_frames):
    for i, j in links:
        tid = track_id_maps[k].get(i)
        if tid is None:
            tid = next_id
            next_id += 1
        track_id_maps[k + 1][j] = tid
n_tracks = next_id
cmap = plt.get_cmap("tab20")
track_colors = {tid: cmap(tid % 20) for tid in range(n_tracks)}

fig, axes = plt.subplots(1, len(frames), figsize=(6 * len(frames), 6), sharey=True)
if len(frames) == 1:
    axes = [axes]

for ax, fr in zip(axes, frames):
    xs, ys, u, v = fr["xs"], fr["ys"], fr["u"], fr["v"]
    speed = np.sqrt(u ** 2 + v ** 2)
    strm = ax.streamplot(xs, ys, u, v, color=speed, cmap="viridis",
                          density=1.4, linewidth=0.7, arrowsize=0.8)
    pts, types = fr["pts"], fr["types"]
    for typ in COLORS:
        mask = [t == typ for t in types]
        if any(mask):
            ax.scatter(pts[mask, 0], pts[mask, 1], c=COLORS[typ], s=45,
                       edgecolors="black", linewidths=0.6, zorder=5, label=typ)
    ax.set_title(f"t = {fr['t']}")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect("equal")

# draw partial-OT tracking links: arrows from frame k critical points to frame k+1
for k, links in enumerate(link_frames):
    axL, axR = axes[k], axes[k + 1]
    P0, P1 = frames[k]["pts"], frames[k + 1]["pts"]
    for i, j in links:
        tid = track_id_maps[k][i]
        col = track_colors[tid]
        con = matplotlib.patches.ConnectionPatch(
            xyA=P0[i], coordsA=axL.transData,
            xyB=P1[j], coordsB=axR.transData,
            color=col, linewidth=1.8, linestyle="--", zorder=10)
        fig.add_artist(con)

legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                        markeredgecolor="black", markersize=8, label=k)
                 for k, c in COLORS.items()]
fig.legend(handles=legend_elems, loc="lower center", ncol=len(COLORS),
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Time-varying vector field critical points tracked with "
             "partial optimal transport (dashed lines = tracks)")
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
out_path = "/workspace/cylinder_critical_point_tracking.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
