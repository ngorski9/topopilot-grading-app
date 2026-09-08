"""
Track critical points of a time-varying 2D vector field (cylinder wake dataset)
across 3 time steps using partial optimal transport, and visualize the tracked
points together with the vector field.

Pipeline:
  1. Load each .vti timestep (arrays 'u', 'v' are the piecewise-linear vector
     field components).
  2. Build the velocity-magnitude scalar field |V| = sqrt(u^2+v^2).
  3. Use TTK's ttkScalarFieldCriticalPoints on |V| to extract minima/maxima/
     saddles. Points with |V| == 0 belong to the masked-out solid cylinder
     body (not real flow features) and are discarded.
  4. Match critical points between consecutive time steps with partial
     optimal transport (POT's partial Wasserstein / partial_wasserstein),
     which -- unlike full optimal transport -- allows points to appear or
     disappear between frames (unbalanced correspondence), as is typical for
     critical point tracking in an unsteady flow.
  5. Plot the vector field (as a subsampled quiver) for each time step with
     the tracked critical points overlaid and connected by trajectory lines.
"""
import numpy as np
import vtk
from vtk.util import numpy_support as ns
import topologytoolkit as ttk_
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/workspace/cylinder"
TIMESTEPS = [1, 2, 3]  # 3 consecutive time steps to track across

CRIT_TYPE_NAMES = {0: "minimum", 1: "saddle", 2: "saddle", 3: "maximum"}
CRIT_TYPE_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:orange", 3: "tab:red"}


def load_timestep(t):
    fname = f"{DATA_DIR}/cylinder{t}.vti"
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()
    data = reader.GetOutput()

    u = ns.vtk_to_numpy(data.GetPointData().GetArray("u")).astype(np.float64)
    v = ns.vtk_to_numpy(data.GetPointData().GetArray("v")).astype(np.float64)
    mag = np.sqrt(u ** 2 + v ** 2)

    mag_arr = ns.numpy_to_vtk(mag, deep=True)
    mag_arr.SetName("mag")
    data.GetPointData().AddArray(mag_arr)

    dims = data.GetDimensions()
    return data, u, v, mag, dims


PERSISTENCE_THRESHOLD = 0.1  # absolute threshold on |V|, prunes noisy/insignificant pairs


def extract_critical_points(data):
    # Persistence-based simplification: removes low-persistence critical
    # point pairs (numerical noise) so only salient flow features (vortex
    # cores, saddles) remain before tracking.
    simplify = ttk_.ttkTopologicalSimplificationByPersistence()
    simplify.SetInputData(data)
    simplify.SetInputArrayToProcess(0, 0, 0, 0, "mag")
    simplify.SetPersistenceThreshold(PERSISTENCE_THRESHOLD)
    simplify.SetThresholdIsAbsolute(True)
    simplify.Update()
    simplified = simplify.GetOutput()

    cp = ttk_.ttkScalarFieldCriticalPoints()
    cp.SetInputData(simplified)
    cp.SetInputArrayToProcess(0, 0, 0, 0, "mag")
    cp.Update()
    out = cp.GetOutput()

    pts = ns.vtk_to_numpy(out.GetPoints().GetData())[:, :2]
    ctype = ns.vtk_to_numpy(out.GetPointData().GetArray("CriticalType"))
    mag = ns.vtk_to_numpy(out.GetPointData().GetArray("mag"))

    # Discard points on the masked-out solid cylinder body (exact-zero
    # velocity magnitude plateau) -- these are not genuine flow features.
    keep = mag > 0.0
    return pts[keep], ctype[keep], mag[keep]


def partial_ot_match(pts_a, pts_b, keep_fraction=0.85):
    """Match critical points between two frames with partial optimal
    transport, allowing unmatched (appearing/disappearing) points."""
    na, nb = len(pts_a), len(pts_b)
    if na == 0 or nb == 0:
        return []

    cost = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)

    a = np.ones(na) / na
    b = np.ones(nb) / nb
    m = keep_fraction * min(a.sum(), b.sum())

    gamma = ot.partial.partial_wasserstein(a, b, cost, m=m)

    matches = []
    for i in range(na):
        j = np.argmax(gamma[i])
        if gamma[i, j] > 1e-12:
            matches.append((i, j, gamma[i, j]))
    return matches


def main():
    frames = []
    for t in TIMESTEPS:
        data, u, v, mag, dims = load_timestep(t)
        pts, ctype, cmag = extract_critical_points(data)
        frames.append(dict(t=t, u=u, v=v, mag=mag, dims=dims, pts=pts,
                            ctype=ctype, cmag=cmag))
        print(f"t={t}: {len(pts)} critical points "
              f"(mins={np.sum(ctype==0)}, saddles={np.sum((ctype==1)|(ctype==2))}, "
              f"maxs={np.sum(ctype==3)})")

    # Match consecutive frames via partial optimal transport.
    tracks = []  # list of (matches_01, matches_12)
    for k in range(len(frames) - 1):
        m = partial_ot_match(frames[k]["pts"], frames[k + 1]["pts"])
        tracks.append(m)
        print(f"t={frames[k]['t']}->t={frames[k+1]['t']}: {len(m)} matches")

    # ---- Visualization ----
    dims = frames[0]["dims"]
    nx, ny = dims[0], dims[1]
    xs = np.arange(nx)
    ys = np.arange(ny)
    step = 6  # subsample for quiver readability

    fig, axes = plt.subplots(1, len(frames), figsize=(6 * len(frames), 6 * ny / nx))

    for k, fr in enumerate(frames):
        ax = axes[k]
        U = fr["u"].reshape(ny, nx)
        V = fr["v"].reshape(ny, nx)
        M = fr["mag"].reshape(ny, nx)

        ax.imshow(M, origin="lower", extent=[0, nx, 0, ny], cmap="Greys", alpha=0.6)
        ax.quiver(xs[::step], ys[::step],
                   U[::step, ::step], V[::step, ::step],
                   color="steelblue", scale=25, width=0.002)

        for cls in np.unique(fr["ctype"]):
            sel = fr["ctype"] == cls
            ax.scatter(fr["pts"][sel, 0], fr["pts"][sel, 1],
                       c=CRIT_TYPE_COLORS.get(cls, "k"), s=45,
                       edgecolors="k", zorder=5,
                       label=CRIT_TYPE_NAMES.get(cls, str(cls)))

        ax.set_title(f"t = {fr['t']}")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("equal")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Draw tracking lines: for each matched pair between consecutive frames,
    # draw a line segment split across the two subplots' shared figure
    # coordinates using ConnectionPatch.
    from matplotlib.patches import ConnectionPatch
    for k, matches in enumerate(tracks):
        pa = frames[k]["pts"]
        pb = frames[k + 1]["pts"]
        for i, j, w in matches:
            con = ConnectionPatch(
                xyA=pa[i], coordsA=axes[k].transData,
                xyB=pb[j], coordsB=axes[k + 1].transData,
                color="lime", linewidth=1.5 + 3 * w, alpha=0.9, zorder=10,
            )
            fig.add_artist(con)

    fig.suptitle("Critical point tracking (partial optimal transport) over 3 time steps\n"
                 "cylinder wake vector field (u, v)", fontsize=13)
    fig.tight_layout()
    out_path = "/workspace/cylinder_critical_point_tracking.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
