"""
Critical point extraction and tracking for the time-varying 'cylinder' PL vector
field dataset (components 'u','v' stored on a 2D regular grid, one .vti file
per time step).

Pipeline
--------
1. Read 3 consecutive time steps (cylinder1/2/3.vti) with VTK.
2. For each time step, extract the critical points of the piecewise-linear
   (bilinearly-interpolated-then-triangulated) vector field (u,v): every grid
   cell (pixel) is split into two triangles, and on each triangle we solve for
   the barycentric coordinates at which the linear interpolant of (u,v)
   vanishes. If the barycentric coordinates lie in the simplex, that location
   is a critical point. The local Jacobian (constant per triangle since the
   field is linear on it) gives the type (sink / source / saddle / center).
3. Track critical points between consecutive time steps with *partial*
   optimal transport (POT's `ot.partial.partial_wasserstein`), which lets
   points appear/disappear (critical points are born/annihilated) instead of
   forcing a perfect assignment.
4. Visualize the original vector field (via matplotlib streamlines, sampled
   from the regular grid) together with the extracted critical points and the
   tracks linking them across the 3 time steps.
"""
import numpy as np
import vtk
from vtk.util import numpy_support as vns
import ot

DATA_FILES = ["./cylinder/cylinder1.vti", "./cylinder/cylinder2.vti", "./cylinder/cylinder3.vti"]


def read_field(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    img = reader.GetOutput()
    dims = img.GetDimensions()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    nx, ny = dims[0], dims[1]
    u = vns.vtk_to_numpy(img.GetPointData().GetArray("u")).reshape(ny, nx)
    v = vns.vtk_to_numpy(img.GetPointData().GetArray("v")).reshape(ny, nx)
    xs = origin[0] + spacing[0] * np.arange(nx)
    ys = origin[1] + spacing[1] * np.arange(ny)
    return xs, ys, u, v


def solve_triangle(p0, p1, p2, v0, v1, v2):
    """Return barycentric-interior zero of the linear interpolant of a 2D
    vector field over triangle (p0,p1,p2) with vertex vectors (v0,v1,v2),
    or None. Also returns the (constant) Jacobian on the triangle."""
    # a*v0 + b*v1 + (1-a-b)*v2 = 0  =>  a*(v0-v2) + b*(v1-v2) = -v2
    M = np.array([[v0[0] - v2[0], v1[0] - v2[0]],
                  [v0[1] - v2[1], v1[1] - v2[1]]])
    det = np.linalg.det(M)
    if abs(det) < 1e-14:
        return None, None
    a, b = np.linalg.solve(M, -np.array(v2))
    c = 1 - a - b
    eps = -1e-9
    if a < eps or b < eps or c < eps:
        return None, None
    pos = a * np.array(p0) + b * np.array(p1) + c * np.array(p2)

    # Jacobian of the PL vector field on this triangle: solve for constant
    # gradients (du/dx,du/dy) and (dv/dx,dv/dy) from the 3 vertex samples.
    A = np.array([[p0[0], p0[1], 1], [p1[0], p1[1], 1], [p2[0], p2[1], 1]])
    ucoef = np.linalg.solve(A, np.array([v0[0], v1[0], v2[0]]))
    vcoef = np.linalg.solve(A, np.array([v0[1], v1[1], v2[1]]))
    J = np.array([[ucoef[0], ucoef[1]], [vcoef[0], vcoef[1]]])
    return pos, J


def classify(J):
    tr = np.trace(J)
    det = np.linalg.det(J)
    disc = tr * tr - 4 * det
    if det < 0:
        return "saddle"
    if disc >= 0:
        return "source" if tr > 0 else "sink"
    return "unstable_focus" if tr > 0 else "stable_focus"


def extract_critical_points(xs, ys, u, v):
    ny, nx = u.shape
    pts, types = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p00 = (xs[i], ys[j]); p10 = (xs[i + 1], ys[j])
            p01 = (xs[i], ys[j + 1]); p11 = (xs[i + 1], ys[j + 1])
            v00 = (u[j, i], v[j, i]); v10 = (u[j, i + 1], v[j, i + 1])
            v01 = (u[j + 1, i], v[j + 1, i]); v11 = (u[j + 1, i + 1], v[j + 1, i + 1])
            for (a, b, c, va, vb, vc) in [
                (p00, p10, p11, v00, v10, v11),
                (p00, p11, p01, v00, v11, v01),
            ]:
                # Skip triangles touching the blanked cylinder body (exactly
                # zero vectors there are a masking artifact, not a genuine
                # critical point).
                if any(abs(vx[0]) < 1e-12 and abs(vx[1]) < 1e-12 for vx in (va, vb, vc)):
                    continue
                pos, J = solve_triangle(a, b, c, va, vb, vc)
                if pos is not None:
                    pts.append(pos)
                    types.append(classify(J))
    return np.array(pts) if pts else np.zeros((0, 2)), types


def partial_ot_match(P0, P1, reg_m_frac=0.9):
    """Match two point sets with partial optimal transport (POT), allowing
    unmatched points (births/deaths). Returns list of (i, j) index pairs."""
    n0, n1 = len(P0), len(P1)
    if n0 == 0 or n1 == 0:
        return []
    C = ot.dist(P0, P1, metric="sqeuclidean")
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    m = reg_m_frac * min(a.sum(), b.sum())
    G = ot.partial.partial_wasserstein(a, b, C, m=m)
    pairs = []
    thresh = G.max() * 1e-6 if G.max() > 0 else 0
    for i in range(n0):
        j = np.argmax(G[i])
        if G[i, j] > thresh:
            pairs.append((i, j))
    return pairs


def main():
    steps = [read_field(f) for f in DATA_FILES]
    cps = []
    for xs, ys, u, v in steps:
        pts, types = extract_critical_points(xs, ys, u, v)
        cps.append((pts, types))
        print(f"time step: {len(pts)} critical points -> "
              f"{ {t: types.count(t) for t in set(types)} }")

    matches_01 = partial_ot_match(cps[0][0], cps[1][0])
    matches_12 = partial_ot_match(cps[1][0], cps[2][0])
    print(f"matches t0->t1: {len(matches_01)} / t1->t2: {len(matches_12)}")

    # Build track ids: chain matches across the 3 steps.
    track_id_t0 = list(range(len(cps[0][0])))
    next_id = len(track_id_t0)
    id_t1 = [None] * len(cps[1][0])
    for i, j in matches_01:
        id_t1[j] = track_id_t0[i]
    for j in range(len(id_t1)):
        if id_t1[j] is None:
            id_t1[j] = next_id
            next_id += 1
    id_t2 = [None] * len(cps[2][0])
    for i, j in matches_12:
        id_t2[j] = id_t1[i]
    for j in range(len(id_t2)):
        if id_t2[j] is None:
            id_t2[j] = next_id
            next_id += 1

    np.savez("cylinder_tracks.npz",
             pts0=cps[0][0], pts1=cps[1][0], pts2=cps[2][0],
             ids0=np.array(track_id_t0), ids1=np.array(id_t1), ids2=np.array(id_t2),
             types0=np.array(cps[0][1], dtype=object),
             types1=np.array(cps[1][1], dtype=object),
             types2=np.array(cps[2][1], dtype=object))

    visualize(steps, cps, [track_id_t0, id_t1, id_t2])


def visualize(steps, cps, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab20")

    for k, ax in enumerate(axes):
        xs, ys, u, v = steps[k]
        speed = np.sqrt(u ** 2 + v ** 2)
        ax.streamplot(xs, ys, u, v, density=1.4, color="0.6", linewidth=0.6, arrowsize=0.8)
        pts, types = cps[k]
        tids = ids[k]
        if len(pts):
            colors = [cmap(t % 20) for t in tids]
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=45, edgecolors="k", zorder=5)
            for (x, y), t in zip(pts, tids):
                ax.annotate(str(t), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"cylinder{k + 1}.vti")
        ax.set_aspect("equal")

    # Draw the tracks (lines connecting matched critical points across steps)
    for ax_pair, (k0, k1) in zip([(axes[0], axes[1])], [(0, 1)]):
        pass

    plt.suptitle("Critical points of the (u,v) vector field tracked over 3 time steps\n"
                 "(partial optimal transport matching, color = track id)")
    plt.tight_layout()
    plt.savefig("cylinder_critical_points.png", dpi=160)
    print("Saved cylinder_critical_points.png")

    # Combined trajectory plot: overlay all 3 steps on one axis with lines
    # connecting the same track id between consecutive steps.
    fig2, ax = plt.subplots(figsize=(7, 12))
    xs, ys, u, v = steps[0]
    ax.streamplot(xs, ys, u, v, density=1.4, color="0.85", linewidth=0.6, arrowsize=0.8)
    id2pt = [dict(zip(ids[k], cps[k][0])) for k in range(3)]
    all_ids = set(ids[0]) | set(ids[1]) | set(ids[2])
    for tid in all_ids:
        chain = [id2pt[k][tid] for k in range(3) if tid in id2pt[k]]
        if len(chain) > 1:
            chain = np.array(chain)
            ax.plot(chain[:, 0], chain[:, 1], "-o", color=cmap(tid % 20), markersize=4)
        elif len(chain) == 1:
            ax.plot(*chain[0], "x", color=cmap(tid % 20), markersize=6)
    ax.set_aspect("equal")
    ax.set_title("Tracked critical point trajectories over time (t1->t2->t3)")
    plt.tight_layout()
    plt.savefig("cylinder_tracks.png", dpi=160)
    print("Saved cylinder_tracks.png")


if __name__ == "__main__":
    main()
