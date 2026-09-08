"""Extract, partially transport, and visualize critical points in cylinder VTI data."""
from pathlib import Path

import numpy as np
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cylinder"
FILES = [DATA / f"cylinder{i}.vti" for i in (1, 2, 3)]


def read_field(path):
    r = vtkXMLImageDataReader(); r.SetFileName(str(path)); r.Update()
    image = r.GetOutput(); dims = image.GetDimensions()
    # VTK point ordering is x fastest; numpy image indexing here is [y, x].
    u = vtk_to_numpy(image.GetPointData().GetArray("u")).reshape(dims[1], dims[0])
    v = vtk_to_numpy(image.GetPointData().GetArray("v")).reshape(dims[1], dims[0])
    return u, v


def critical_points(u, v):
    """Zeros of a bilinear field, found with Newton iterations per sign-changing cell."""
    pts, kinds = [], []
    h, w = u.shape
    for j in range(h - 1):
        for i in range(w - 1):
            U = u[j:j+2, i:i+2]; V = v[j:j+2, i:i+2]
            # The solid cylinder is encoded as exact zero velocity.  Its boundary
            # creates whole strings of non-isolated roots, which are not critical
            # points of the surrounding piecewise-linear flow.
            if np.any(np.hypot(U, V) < 1e-12):
                continue
            # A component must bracket zero in a cell for a common zero to exist.
            if U.min() > 0 or U.max() < 0 or V.min() > 0 or V.max() < 0:
                continue
            # coefficients f=a+b*x+c*y+d*x*y, with local x,y in [0,1]
            def coef(A): return A[0,0], A[0,1]-A[0,0], A[1,0]-A[0,0], A[1,1]-A[0,1]-A[1,0]+A[0,0]
            au, av = coef(U), coef(V)
            x = y = 0.5
            for _ in range(16):
                fu = au[0]+au[1]*x+au[2]*y+au[3]*x*y
                fv = av[0]+av[1]*x+av[2]*y+av[3]*x*y
                J = np.array([[au[1]+au[3]*y, au[2]+au[3]*x],
                              [av[1]+av[3]*y, av[2]+av[3]*x]])
                try: dxy = np.linalg.solve(J, [-fu, -fv])
                except np.linalg.LinAlgError: break
                x, y = x + dxy[0], y + dxy[1]
                if np.linalg.norm(dxy) < 1e-10: break
            if -1e-8 <= x <= 1+1e-8 and -1e-8 <= y <= 1+1e-8:
                jac = np.array([[au[1]+au[3]*y, au[2]+au[3]*x],
                                [av[1]+av[3]*y, av[2]+av[3]*x]])
                eig = np.linalg.eigvals(jac)
                kinds.append("saddle" if np.linalg.det(jac) < 0 else ("source" if np.trace(jac) > 0 else "sink"))
                pts.append((i+x, j+y))
    return np.asarray(pts), np.asarray(kinds)


fields = [read_field(f) for f in FILES]
cps = [critical_points(*f) for f in fields]
for t, (p, k) in enumerate(cps, 1):
    print(f"t={t}: {len(p)} critical points; " + ", ".join(f"{z}={np.count_nonzero(k==z)}" for z in ('saddle','source','sink')))

# Partial OT transports 85% of the uniform mass, allowing short-lived points to be unmatched.
links = []
for t in range(2):
    p, q = cps[t][0], cps[t+1][0]
    a, b = np.full(len(p), 1/len(p)), np.full(len(q), 1/len(q))
    C = ot.dist(p, q, metric="sqeuclidean") / (150.0**2 + 450.0**2)
    G = ot.partial.partial_wasserstein(a, b, C, m=0.85)
    # Each row's dominant transported mass defines its correspondence.
    for i in range(len(p)):
        j = G[i].argmax()
        if G[i, j] > 1e-9:
            links.append((t, i, j, G[i, j]))
print("partial-OT links:", len(links), "(transported mass 0.85 at each transition)")

colors = {'saddle':'#e63946', 'source':'#1565c0', 'sink':'#2a9d8f'}
fig, axes = plt.subplots(1, 3, figsize=(18, 10), constrained_layout=True, sharex=True, sharey=True)
for t, (ax, (u, v), (p, k)) in enumerate(zip(axes, fields, cps)):
    yy, xx = np.mgrid[0:u.shape[0]:12, 0:u.shape[1]:12]
    ax.quiver(xx, yy, u[::12,::12], v[::12,::12], color="#77838d", alpha=.62,
              pivot="mid", scale=2.3, width=.0022)
    for kind in colors:
        m = k == kind
        ax.scatter(p[m,0], p[m,1], s=75, c=colors[kind], edgecolors="white", linewidths=.9,
                   label=kind if t == 0 else None, zorder=4)
    # Draw each incoming OT link on the destination field, so motion is visible in context.
    if t:
        for lt, i, j, mass in links:
            if lt == t-1:
                a, b = cps[lt][0][i], cps[t][0][j]
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#202124", lw=1.3, alpha=.72))
    ax.set_title(f"Cylinder vector field — time step {t+1}")
    ax.set_aspect("equal"); ax.set_xlim(0, 149); ax.set_ylim(449, 0)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(loc="lower right", title="Critical point type")
fig.suptitle("Critical points tracked using partial optimal transport (85% mass)", fontsize=16)
out = ROOT / "cylinder_critical_points_partial_ot.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("wrote", out)
