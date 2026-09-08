#!/opt/conda/bin/pvpython
"""Extract and partially-optimally-transport critical points in cylinder VTI data.

Run with: /opt/conda/bin/pvpython track_cylinder_critical_points.py
"""
from pathlib import Path
import csv
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cylinder"
OUT = ROOT / "cylinder_tracking_output"
OUT.mkdir(exist_ok=True)
STEPS = [1, 2, 3]


def read_field(step):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(DATA / f"cylinder{step}.vti"))
    reader.Update()
    image = reader.GetOutput()
    nx, ny, _ = image.GetDimensions()
    # VTK point ordering is x-fastest, so transpose to [y, x] for plotting.
    u = vtk_to_numpy(image.GetPointData().GetArray("u")).reshape(ny, nx)
    v = vtk_to_numpy(image.GetPointData().GetArray("v")).reshape(ny, nx)
    return u, v


def bilinear_root(u, v, x, y):
    """Return a zero inside one grid cell, using Newton on bilinear components."""
    # F(xi,eta) coefficients: a + b xi + c eta + d xi eta
    def coeff(a):
        return a[0, 0], a[0, 1] - a[0, 0], a[1, 0] - a[0, 0], a[1, 1] - a[0, 1] - a[1, 0] + a[0, 0]
    cu, cv = coeff(u[y:y+2, x:x+2]), coeff(v[y:y+2, x:x+2])
    q = np.array([0.5, 0.5])
    for _ in range(20):
        xi, eta = q
        f = np.array([cu[0]+cu[1]*xi+cu[2]*eta+cu[3]*xi*eta,
                      cv[0]+cv[1]*xi+cv[2]*eta+cv[3]*xi*eta])
        jac = np.array([[cu[1]+cu[3]*eta, cu[2]+cu[3]*xi],
                        [cv[1]+cv[3]*eta, cv[2]+cv[3]*xi]])
        try:
            d = np.linalg.solve(jac, f)
        except np.linalg.LinAlgError:
            return None
        q -= d
        if np.linalg.norm(d) < 1e-10:
            break
    if np.all(q >= -1e-7) and np.all(q <= 1 + 1e-7):
        xi, eta = q
        jac = np.array([[cu[1]+cu[3]*eta, cu[2]+cu[3]*xi],
                        [cv[1]+cv[3]*eta, cv[2]+cv[3]*xi]])
        eig = np.linalg.eigvals(jac)
        if np.linalg.det(jac) < 0:
            kind = "saddle"
        elif np.iscomplex(eig).any():
            kind = "source" if np.trace(jac) > 0 else "sink"
        else:
            kind = "source" if np.trace(jac) > 0 else "sink"
        return (x + xi, y + eta, kind)
    return None


def critical_points(u, v):
    points = []
    ny, nx = u.shape
    for y in range(ny - 1):
        for x in range(nx - 1):
            uu, vv = u[y:y+2, x:x+2], v[y:y+2, x:x+2]
            # Strict sign changes exclude the zero-valued exterior/mask, whose
            # non-isolated roots are not vector-field critical points.
            if uu.min() < -1e-12 < uu.max() and vv.min() < -1e-12 < vv.max():
                p = bilinear_root(u, v, x, y)
                if p is not None and not any(np.hypot(p[0]-r[0], p[1]-r[1]) < 1e-5 for r in points):
                    points.append(p)
    return points


def partial_match(a, b):
    """Partial OT matching, preserving saddle versus non-saddle topology."""
    matches = []
    for typ in ("saddle", "non_saddle"):
        ia = [i for i, p in enumerate(a) if (p[2] == "saddle") == (typ == "saddle")]
        ib = [i for i, p in enumerate(b) if (p[2] == "saddle") == (typ == "saddle")]
        if not ia or not ib:
            continue
        A = np.array([[a[i][0], a[i][1]] for i in ia])
        B = np.array([[b[i][0], b[i][1]] for i in ib])
        cost = ot.dist(A, B, metric="sqeuclidean")
        # Transport 90% of the smaller measure: explicit partial OT permits
        # unmatched births/deaths rather than forcing implausible assignments.
        mass = 0.90 * min(len(ia), len(ib)) / max(len(ia), len(ib))
        plan = ot.partial.partial_wasserstein(np.ones(len(ia))/len(ia), np.ones(len(ib))/len(ib), cost, m=mass)
        # Discard small fractional residuals in the partial plan.  The
        # remaining edges are its high-confidence, mutually local transport.
        core_mass = 0.5 / max(len(ia), len(ib))
        for r, c in zip(*np.where((plan >= core_mass) & (np.sqrt(cost) <= 5.0))):
            matches.append((ia[r], ib[c], float(plan[r, c]), float(np.sqrt(cost[r, c]))))
    return matches


fields = [read_field(t) for t in STEPS]
points = [critical_points(*f) for f in fields]
matches = [partial_match(points[i], points[i+1]) for i in range(2)]

# Form three-time tracks only when consecutive partial-OT links agree.
tracks = []
for i0, i1, mass01, d01 in matches[0]:
    for j1, i2, mass12, d12 in matches[1]:
        if i1 == j1:
            tracks.append((i0, i1, i2, min(mass01, mass12), d01, d12))

with open(OUT / "critical_points_and_partial_ot.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time_step", "point_id", "x", "y", "type"])
    for t, ps in zip(STEPS, points):
        for i, (x, y, kind) in enumerate(ps): w.writerow([t, i, x, y, kind])
with open(OUT / "partial_ot_tracks.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id_t1", "id_t2", "id_t3", "transport_mass", "distance_1_2", "distance_2_3"])
    w.writerows(tracks)

colors = {"saddle": "#e63946", "source": "#ffb000", "sink": "#4cc9f0"}
fig, axes = plt.subplots(1, 3, figsize=(14, 15), sharex=True, sharey=True)
for ax, (t, (u, v), ps) in zip(axes, zip(STEPS, fields, points)):
    speed = np.hypot(u, v)
    ax.imshow(speed, origin="lower", cmap="Greys", alpha=.68, extent=[0, u.shape[1]-1, 0, u.shape[0]-1])
    # Downsampled arrows retain the original vector field while remaining legible.
    yy, xx = np.mgrid[0:u.shape[0]:10, 0:u.shape[1]:10]
    ax.quiver(xx, yy, u[::10, ::10], v[::10, ::10], color="#245", alpha=.8, scale=25, width=.0025)
    for kind in colors:
        q = [(x, y) for x, y, k in ps if k == kind]
        if q:
            q = np.array(q); ax.scatter(q[:,0], q[:,1], s=74, c=colors[kind], edgecolors="black", linewidths=.7, label=kind)
    ax.set_title(f"Cylinder field — time step {t}")
    ax.set_aspect("equal"); ax.set_xlabel("x")
axes[0].set_ylabel("y")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, title="Critical-point type", bbox_to_anchor=(.5, .99))
fig.subplots_adjust(left=.07, right=.99, bottom=.05, top=.94, wspace=.18)
fig.savefig(OUT / "critical_points_on_vector_fields.png", dpi=220)
plt.close(fig)

# A spatial track view, retaining the time-specific field as a light backdrop.
u, v = fields[1]
fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
ax.imshow(np.hypot(u, v), origin="lower", cmap="Greys", alpha=.55, extent=[0,u.shape[1]-1,0,u.shape[0]-1])
yy, xx = np.mgrid[0:u.shape[0]:10, 0:u.shape[1]:10]
ax.quiver(xx, yy, u[::10,::10], v[::10,::10], color="#245", alpha=.55, scale=25, width=.002)
for n, (i0, i1, i2, mass, _, _) in enumerate(tracks):
    xy = np.array([[points[0][i0][0], points[0][i0][1]], [points[1][i1][0], points[1][i1][1]], [points[2][i2][0], points[2][i2][1]]])
    typ = points[0][i0][2]
    ax.plot(xy[:,0], xy[:,1], "-", color=colors[typ], lw=2.5, alpha=.9)
    ax.scatter(xy[:,0], xy[:,1], c=colors[typ], s=[45,70,95], edgecolors="black", linewidths=.5, zorder=3)
    ax.text(xy[1,0]+2, xy[1,1]+2, f"{n}", fontsize=8)
ax.set(title="Partial-OT critical-point tracks (t=1 → 2 → 3)", xlabel="x", ylabel="y", aspect="equal")
fig.savefig(OUT / "partial_ot_critical_point_tracks.png", dpi=220)
plt.close(fig)

summary = {"time_steps": STEPS, "critical_points_per_step": [len(x) for x in points], "partial_ot_links": [len(x) for x in matches], "three_step_tracks": len(tracks), "partial_mass_fraction": 0.90}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
