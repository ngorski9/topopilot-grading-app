import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Load the asymmetric piecewise-linear tensor field
# ---------------------------------------------------------------------------
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("Ocean.vti")
reader.Update()
img = reader.GetOutput()

nx, ny, nz = img.GetDimensions()
ox, oy, oz = img.GetOrigin()
sx, sy, sz = img.GetSpacing()
pd = img.GetPointData()

def get_field(name):
    arr = vtk_to_numpy(pd.GetArray(name))
    # VTK image data point order is x-fastest
    return arr.reshape(ny, nx).T  # -> shape (nx, ny), [i,j]

A = get_field("A")
B = get_field("B")
C = get_field("C")
D = get_field("D")

xs = ox + sx * np.arange(nx)
ys = oy + sy * np.arange(ny)

ncell_x, ncell_y = nx - 1, ny - 1

# ---------------------------------------------------------------------------
# Deviator / rotation decomposition
#   E = (A-D)/2 , F = (B+C)/2 , W = (C-B)/2
#   eigenvalues real  <=>  Delta = E^2 + F^2 - W^2 > 0
#   degenerate points <=>  E = 0 and F = 0  (classic wedge/trisector points)
# ---------------------------------------------------------------------------
E = 0.5 * (A - D)
F = 0.5 * (B + C)
W = 0.5 * (C - B)

# ---------------------------------------------------------------------------
# Find exact degenerate points on the piecewise-linear (triangulated) field
# ---------------------------------------------------------------------------
def triangle_zero(p0, p1, p2, e0, e1, e2, f0, f1, f2):
    """Solve for barycentric coords where E=F=0 is linearly interpolated
    inside the triangle (p0,p1,p2). Returns (point_xy, jacobian_sign) or None."""
    M = np.array([
        [e0, e1, e2],
        [f0, f1, f2],
        [1.0, 1.0, 1.0],
    ])
    rhs = np.array([0.0, 0.0, 1.0])
    try:
        lam = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    eps = 1e-9
    if np.any(lam < -eps) or np.any(lam > 1 + eps):
        return None
    pt = lam[0] * np.array(p0) + lam[1] * np.array(p1) + lam[2] * np.array(p2)

    # gradient of E and F over the (linear) triangle
    Mxy = np.array([[p1[0] - p0[0], p1[1] - p0[1]],
                     [p2[0] - p0[0], p2[1] - p0[1]]])
    try:
        gradE = np.linalg.solve(Mxy, np.array([e1 - e0, e2 - e0]))
        gradF = np.linalg.solve(Mxy, np.array([f1 - f0, f2 - f0]))
    except np.linalg.LinAlgError:
        return None
    J = gradE[0] * gradF[1] - gradE[1] * gradF[0]
    return pt, J

degenerate_points = []  # (x, y, kind)  kind in {"wedge", "trisector"}

for i in range(ncell_x):
    for j in range(ncell_y):
        p00 = (xs[i], ys[j])
        p10 = (xs[i + 1], ys[j])
        p11 = (xs[i + 1], ys[j + 1])
        p01 = (xs[i], ys[j + 1])

        verts = {
            (0, 0): p00, (1, 0): p10, (1, 1): p11, (0, 1): p01,
        }
        Eval = {(0, 0): E[i, j], (1, 0): E[i + 1, j], (1, 1): E[i + 1, j + 1], (0, 1): E[i, j + 1]}
        Fval = {(0, 0): F[i, j], (1, 0): F[i + 1, j], (1, 1): F[i + 1, j + 1], (0, 1): F[i, j + 1]}

        triangles = [((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))]
        for tri in triangles:
            p0, p1, p2 = (verts[v] for v in tri)
            e0, e1, e2 = (Eval[v] for v in tri)
            f0, f1, f2 = (Fval[v] for v in tri)
            res = triangle_zero(p0, p1, p2, e0, e1, e2, f0, f1, f2)
            if res is None:
                continue
            (px, py), J = res
            kind = "wedge" if J > 0 else "trisector"
            degenerate_points.append((px, py, kind))

print(f"Found {len(degenerate_points)} degenerate points")
n_wedge = sum(1 for p in degenerate_points if p[2] == "wedge")
n_tri = sum(1 for p in degenerate_points if p[2] == "trisector")
print(f"  wedges: {n_wedge}, trisectors: {n_tri}")

# ---------------------------------------------------------------------------
# Rasterize the eigenvector partition at 10 pixels per unit square
# ---------------------------------------------------------------------------
PX_PER_SQUARE = 10
width = ncell_x * PX_PER_SQUARE
height = ncell_y * PX_PER_SQUARE

# pixel-center sample coordinates in data space
px_x = ox + sx * (np.arange(width) + 0.5) / PX_PER_SQUARE
px_y = oy + sy * (np.arange(height) + 0.5) / PX_PER_SQUARE

# bilinear interpolation of A,B,C,D onto the raster grid
gx = (px_x - ox) / sx
gy = (px_y - oy) / sy
i0 = np.clip(np.floor(gx).astype(int), 0, ncell_x - 1)
j0 = np.clip(np.floor(gy).astype(int), 0, ncell_y - 1)
tx = (gx - i0)[:, None]     # (width,1)
ty = (gy - j0)[None, :]     # (1,height)

def bilinear(field):
    f00 = field[i0[:, None], j0[None, :]]
    f10 = field[i0[:, None] + 1, j0[None, :]]
    f01 = field[i0[:, None], j0[None, :] + 1]
    f11 = field[i0[:, None] + 1, j0[None, :] + 1]
    return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty) +
            f01 * (1 - tx) * ty + f11 * tx * ty)

Ar = bilinear(A)
Br = bilinear(B)
Cr = bilinear(C)
Dr = bilinear(D)

Er = 0.5 * (Ar - Dr)
Fr = 0.5 * (Br + Cr)
Wr = 0.5 * (Cr - Br)
Delta = Er ** 2 + Fr ** 2 - Wr ** 2

partition = np.where(Delta >= 0, 1, 0)  # 1 = real domain, 0 = complex domain

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
fig_w, fig_h = width / 100.0, height / 100.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#2166ac", "#f4a582"])  # complex domain, real domain
ax.imshow(partition.T, origin="lower",
          extent=[ox, ox + sx * ncell_x, oy, oy + sy * ncell_y],
          cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

radius = 1.0
for (x, y, kind) in degenerate_points:
    color = "pink" if kind == "trisector" else "white"
    circ = Circle((x, y), radius, facecolor=color, edgecolor="black", linewidth=0.5, zorder=5)
    ax.add_patch(circ)

ax.set_xlim(ox, ox + sx * ncell_x)
ax.set_ylim(oy, oy + sy * ncell_y)
ax.set_aspect("equal")
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("Ocean_eigenvector_partition.png", dpi=100)
print("Saved Ocean_eigenvector_partition.png")
