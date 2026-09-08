"""
Track critical points of the time-varying piecewise-linear vector field
(component arrays 'u','v') in ./cylinder across 3 time steps, matching
critical points between consecutive steps with partial optimal transport
(POT library), and visualize the tracks over the original vector field.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TIMESTEPS = [1, 2, 3]
DATA_DIR = "/workspace/cylinder"


def load_field(t):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(f"{DATA_DIR}/cylinder{t}.vti")
    reader.Update()
    img = reader.GetOutput()
    dims = img.GetDimensions()  # (nx, ny, 1)
    nx, ny = dims[0], dims[1]
    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(pd.GetArray("v")).reshape(ny, nx)
    ox, oy, _ = img.GetOrigin()
    sx, sy, _ = img.GetSpacing()
    xs = ox + sx * np.arange(nx)
    ys = oy + sy * np.arange(ny)
    return xs, ys, u, v


def bary_weights_for_zero(p0, p1, p2):
    """Solve for barycentric weights (w0,w1,w2>=0, sum=1) such that
    w0*p0 + w1*p1 + w2*p2 = 0, where p_i are 2D vector values at triangle
    vertices. Returns None if no such non-negative solution exists
    (i.e. the origin is not in the convex hull of p0,p1,p2)."""
    A = np.array([
        [p0[0], p1[0], p2[0]],
        [p0[1], p1[1], p2[1]],
        [1.0, 1.0, 1.0],
    ])
    b = np.array([0.0, 0.0, 1.0])
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if np.all(w >= -1e-9):
        return w
    return None


def classify(jac):
    """Classify a 2x2 Jacobian of the PL vector field on the triangle."""
    det = np.linalg.det(jac)
    tr = np.trace(jac)
    if det < 0:
        return "saddle"
    if tr > 0:
        return "source"
    return "sink"


def extract_critical_points(xs, ys, u, v):
    """Extract critical points of the PL vector field obtained by splitting
    each grid cell into 2 triangles and linearly interpolating (u,v)."""
    ny, nx = u.shape
    pts = []
    types = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            corners_xy = {
                "00": (x0, y0), "10": (x1, y0),
                "01": (x0, y1), "11": (x1, y1),
            }
            corners_uv = {
                "00": (u[j, i], v[j, i]), "10": (u[j, i + 1], v[j, i + 1]),
                "01": (u[j + 1, i], v[j + 1, i]), "11": (u[j + 1, i + 1], v[j + 1, i + 1]),
            }
            # split quad into two triangles: (00,10,11) and (00,11,01)
            for tri in (("00", "10", "11"), ("00", "11", "01")):
                pv = [corners_uv[k] for k in tri]
                pp = [corners_xy[k] for k in tri]
                # skip degenerate/blanked cells (e.g. inside the solid
                # cylinder, where velocity is identically zero) -- these
                # are not genuine isolated critical points
                if min(np.hypot(*pv[0]), np.hypot(*pv[1]), np.hypot(*pv[2])) < 1e-6:
                    continue
                w = bary_weights_for_zero(*pv)
                if w is None:
                    continue
                pos = w[0] * np.array(pp[0]) + w[1] * np.array(pp[1]) + w[2] * np.array(pp[2])
                # local jacobian via affine fit on triangle (constant per triangle)
                P = np.array([[pp[1][0] - pp[0][0], pp[2][0] - pp[0][0]],
                              [pp[1][1] - pp[0][1], pp[2][1] - pp[0][1]]])
                Fv = np.array([[pv[1][0] - pv[0][0], pv[2][0] - pv[0][0]],
                               [pv[1][1] - pv[0][1], pv[2][1] - pv[0][1]]])
                try:
                    jac = Fv @ np.linalg.inv(P)
                except np.linalg.LinAlgError:
                    continue
                pts.append(pos)
                types.append(classify(jac))
    return np.array(pts), np.array(types)


def partial_ot_match(P0, P1, trim=0.15):
    """Match critical points between two point sets using partial optimal
    transport (unbalanced masses = allow birth/death), based on POT."""
    n0, n1 = len(P0), len(P1)
    if n0 == 0 or n1 == 0:
        return []
    C = ot.dist(P0, P1, metric="euclidean")
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    # mass to be transported: as much as possible, minus a trim fraction
    # reserved for points that are born/die (no match) between steps
    m = (1 - trim) * min(a.sum(), b.sum())
    G = ot.partial.partial_wasserstein(a, b, C, m=m)
    matches = []
    thresh = G.max() * 0.05 if G.max() > 0 else 0
    for i in range(n0):
        j = np.argmax(G[i])
        if G[i, j] > thresh and G[i, j] > 0:
            matches.append((i, j, G[i, j]))
    return matches


def main():
    fields = {t: load_field(t) for t in TIMESTEPS}
    crit = {}
    for t in TIMESTEPS:
        xs, ys, u, v = fields[t]
        pts, types = extract_critical_points(xs, ys, u, v)
        crit[t] = (pts, types)
        print(f"t={t}: {len(pts)} critical points "
              f"(sources={np.sum(types=='source')}, sinks={np.sum(types=='sink')}, "
              f"saddles={np.sum(types=='saddle')})")

    # match consecutive timesteps with partial optimal transport
    tracks = []  # list of (t, idx0, idx1) match between t and t+1
    for a, b in zip(TIMESTEPS[:-1], TIMESTEPS[1:]):
        P0, T0 = crit[a]
        P1, T1 = crit[b]
        matches = partial_ot_match(P0, P1)
        tracks.append((a, b, matches))
        print(f"matched {len(matches)} critical points between t={a} and t={b}")

    # ---- visualization ----
    fig, ax = plt.subplots(figsize=(6, 14))
    xs, ys, u, v = fields[TIMESTEPS[0]]
    step = 4
    Xs, Ys = np.meshgrid(xs[::step], ys[::step])
    Us, Vs = u[::step, ::step], v[::step, ::step]
    mag = np.hypot(Us, Vs)
    mag[mag == 0] = 1.0
    ax.quiver(Xs, Ys, Us / mag, Vs / mag, mag, cmap="Greys", scale=40,
              width=0.0025, zorder=1)

    colors = {"source": "tab:red", "sink": "tab:blue", "saddle": "tab:green"}
    cmap_t = ["black", "purple", "orange"]
    for i, t in enumerate(TIMESTEPS):
        pts, types = crit[t]
        for typ in colors:
            mask = types == typ
            if mask.any():
                ax.scatter(pts[mask, 0], pts[mask, 1], s=25, c=colors[typ],
                           marker="o" if i == 0 else ("s" if i == 1 else "^"),
                           edgecolors=cmap_t[i], linewidths=0.8,
                           label=f"t={t} {typ}", zorder=3)

    # draw trajectories by chaining matches across all steps
    positions_by_t = {t: crit[t][0] for t in TIMESTEPS}
    segments = []
    for a, b, matches in tracks:
        for i, j, w in matches:
            p0 = positions_by_t[a][i]
            p1 = positions_by_t[b][j]
            segments.append([p0, p1])
    if segments:
        lc = LineCollection(segments, colors="black", linewidths=1.5, alpha=0.8, zorder=2)
        ax.add_collection(lc)

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect("equal")
    ax.set_title("Critical point tracking (partial optimal transport)\n"
                  "over vector field (u,v), t=1,2,3")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("/workspace/cylinder_tracking.png", dpi=150)
    print("saved /workspace/cylinder_tracking.png")


if __name__ == "__main__":
    main()
