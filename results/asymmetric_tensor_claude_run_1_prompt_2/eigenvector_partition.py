import vtk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from vtk.util.numpy_support import vtk_to_numpy

# ---- Load data ----
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("Ocean.vti")
reader.Update()
img = reader.GetOutput()
nx, ny, nz = img.GetDimensions()
ox, oy, oz = img.GetOrigin()
sx, sy, sz = img.GetSpacing()

pd = img.GetPointData()
def get_arr(name):
    a = vtk_to_numpy(pd.GetArray(name)).astype(np.float64)
    # VTK point ordering: x varies fastest
    return a.reshape(ny, nx)

A = get_arr("A")
B = get_arr("B")
C = get_arr("C")
D = get_arr("D")

# tensor T = [[A, B], [C, D]]
delta = 0.5 * (A - D)          # symmetric deviator, diagonal part
beta  = 0.5 * (B + C)          # symmetric deviator, off-diagonal part
gamma = 0.5 * (B - C)          # antisymmetric part
Disc  = delta**2 + beta**2 - gamma**2   # discriminant: >0 real distinct eigenvalues, <0 complex

# coordinates of grid points
xs = ox + sx * np.arange(nx)
ys = oy + sy * np.arange(ny)

# ---- Find degenerate points (zeros of (delta, beta) field) per triangulated cell ----
degenerate_points = []  # (x, y, type)  type: 'wedge' or 'trisector'

def solve_and_classify(p1, p2, p3, d1, d2, d3, b1, b2, b3):
    # linear system for barycentric coords (L2, L3), L1 = 1-L2-L3
    m00 = d2 - d1
    m01 = d3 - d1
    m10 = b2 - b1
    m11 = b3 - b1
    det = m00 * m11 - m01 * m10
    if det == 0:
        return None
    rhs0 = -d1
    rhs1 = -b1
    L2 = (m11 * rhs0 - m01 * rhs1) / det
    L3 = (-m10 * rhs0 + m00 * rhs1) / det
    L1 = 1.0 - L2 - L3
    eps = -1e-9
    if L1 < eps or L2 < eps or L3 < eps:
        return None
    L1 = min(max(L1, 0.0), 1.0)
    L2 = min(max(L2, 0.0), 1.0)
    L3 = min(max(L3, 0.0), 1.0)
    x = L1 * p1[0] + L2 * p2[0] + L3 * p3[0]
    y = L1 * p1[1] + L2 * p2[1] + L3 * p3[1]
    kind = "wedge" if det > 0 else "trisector"
    return (x, y, kind)

for j in range(ny - 1):
    for i in range(nx - 1):
        # cell corners
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[j], ys[j + 1]
        p00 = (x0, y0); p10 = (x1, y0); p01 = (x0, y1); p11 = (x1, y1)
        d00, d10, d01, d11 = delta[j, i], delta[j, i+1], delta[j+1, i], delta[j+1, i+1]
        b00, b10, b01, b11 = beta[j, i], beta[j, i+1], beta[j+1, i], beta[j+1, i+1]

        # triangle 1: p00, p10, p11
        r = solve_and_classify(p00, p10, p11, d00, d10, d11, b00, b10, b11)
        if r:
            degenerate_points.append(r)
        # triangle 2: p00, p11, p01
        r = solve_and_classify(p00, p11, p01, d00, d11, d01, b00, b11, b01)
        if r:
            degenerate_points.append(r)

print(f"Found {len(degenerate_points)} degenerate points")
n_wedge = sum(1 for p in degenerate_points if p[2] == "wedge")
n_tri = sum(1 for p in degenerate_points if p[2] == "trisector")
print(f"  wedges: {n_wedge}, trisectors: {n_tri}")

# ---- Render ----
px_per_unit = 10
width_units = xs[-1] - xs[0]
height_units = ys[-1] - ys[0]
fig_w = width_units * px_per_unit / 100.0
fig_h = height_units * px_per_unit / 100.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

# eigenvector partition: real domain (Disc > 0) vs complex domain (Disc < 0)
partition = np.where(Disc > 0, 1.0, 0.0)
cmap = matplotlib.colors.ListedColormap(["#7fb2d9", "#f2f2e6"])  # complex=blue, real=cream
ax.imshow(partition, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
          cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

radius = 1.0
for (x, y, kind) in degenerate_points:
    if kind == "trisector":
        c = Circle((x, y), radius, facecolor="pink", edgecolor="black", linewidth=0.5, zorder=3)
    else:
        c = Circle((x, y), radius, facecolor="white", edgecolor="black", linewidth=0.5, zorder=3)
    ax.add_patch(c)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])
ax.set_aspect("equal")
ax.axis("off")

out_path = "/workspace/Ocean_eigenvector_partition.png"
plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
print("saved", out_path)
