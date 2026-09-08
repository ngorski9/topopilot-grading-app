import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('Ocean.vti')
reader.Update()
data = reader.GetOutput()
dims = data.GetDimensions()
spacing = data.GetSpacing()
origin = data.GetOrigin()
pd = data.GetPointData()

A = vtk_to_numpy(pd.GetArray('A')).reshape(dims[1], dims[0])
B = vtk_to_numpy(pd.GetArray('B')).reshape(dims[1], dims[0])
C = vtk_to_numpy(pd.GetArray('C')).reshape(dims[1], dims[0])
D = vtk_to_numpy(pd.GetArray('D')).reshape(dims[1], dims[0])

nx, ny = dims[0], dims[1]
xmax = origin[0] + (nx - 1) * spacing[0]
ymax = origin[1] + (ny - 1) * spacing[1]
ncellsx, ncellsy = nx - 1, ny - 1

PPS = 10  # pixels per square (grid cell)

# ---------- degenerate point extraction (piecewise-linear triangulation) ----------
a_field = (A - D) / 2.0
b_field = (B + C) / 2.0

raw_pts = []
raw_types = []
for j in range(ncellsy):
    for i in range(ncellsx):
        tris = [
            [(i, j), (i + 1, j), (i, j + 1)],
            [(i + 1, j), (i + 1, j + 1), (i, j + 1)],
        ]
        for tri in tris:
            (x0, y0), (x1, y1), (x2, y2) = tri
            a0, a1, a2 = a_field[y0, x0], a_field[y1, x1], a_field[y2, x2]
            b0, b1, b2 = b_field[y0, x0], b_field[y1, x1], b_field[y2, x2]
            M = np.array([[a1 - a0, a2 - a0], [b1 - b0, b2 - b0]])
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) < 1e-14:
                continue
            rhs = np.array([-a0, -b0])
            l1, l2 = np.linalg.solve(M, rhs)
            l0 = 1 - l1 - l2
            if min(l0, l1, l2) >= -1e-9 and max(l0, l1, l2) <= 1 + 1e-9:
                x = x0 + l1 * (x1 - x0) + l2 * (x2 - x0)
                y = y0 + l1 * (y1 - y0) + l2 * (y2 - y0)
                Mxy = np.array([[x1 - x0, y1 - y0], [x2 - x0, y2 - y0]])
                grad_a = np.linalg.solve(Mxy, np.array([a1 - a0, a2 - a0]))
                grad_b = np.linalg.solve(Mxy, np.array([b1 - b0, b2 - b0]))
                J = grad_a[0] * grad_b[1] - grad_a[1] * grad_b[0]
                raw_pts.append((x, y))
                raw_types.append(1 if J > 0 else 0)  # 1 = wedge, 0 = trisector

raw_pts = np.array(raw_pts)
raw_types = np.array(raw_types)

# dedupe points shared between adjacent triangles
dedup_pts, dedup_types = [], []
used = np.zeros(len(raw_pts), dtype=bool)
for k in range(len(raw_pts)):
    if used[k]:
        continue
    close = np.where((np.abs(raw_pts[:, 0] - raw_pts[k, 0]) < 1e-6) &
                      (np.abs(raw_pts[:, 1] - raw_pts[k, 1]) < 1e-6))[0]
    used[close] = True
    dedup_pts.append(raw_pts[k])
    dedup_types.append(raw_types[k])
dedup_pts = np.array(dedup_pts)
dedup_types = np.array(dedup_types)

n_wedge = int((dedup_types == 1).sum())
n_tri = int((dedup_types == 0).sum())
print(f"Degenerate points: {len(dedup_pts)}  (trisectors={n_tri}, wedges={n_wedge})")

# ---------- eigenvector partition raster ----------
def bilinear(V, x, y):
    x = np.clip(x, 0, xmax - 1e-9)
    y = np.clip(y, 0, ymax - 1e-9)
    i = np.floor(x).astype(int)
    j = np.floor(y).astype(int)
    fx, fy = x - i, y - j
    v00, v10 = V[j, i], V[j, i + 1]
    v01, v11 = V[j + 1, i], V[j + 1, i + 1]
    return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11

W = ncellsx * PPS
H = ncellsy * PPS
xs = (np.arange(W) + 0.5) / PPS
ys = (np.arange(H) + 0.5) / PPS
X, Y = np.meshgrid(xs, ys)

Ai, Bi, Ci, Di = bilinear(A, X, Y), bilinear(B, X, Y), bilinear(C, X, Y), bilinear(D, X, Y)
a = (Ai - Di) / 2.0
b = (Bi + Ci) / 2.0
r = (Ci - Bi) / 2.0
s = np.sqrt(a ** 2 + b ** 2)
real = s >= np.abs(r)
north = r > 0

label = np.zeros((H, W), dtype=int)
label[(~real) & (~north)] = 0  # W_{c,s}
label[(real) & (~north)] = 1   # W_{r,s}
label[(real) & (north)] = 2    # W_{r,n}
label[(~real) & (north)] = 3   # W_{c,n}

cmap = ListedColormap(['#440154', '#3b528b', '#21908d', '#fde725'])

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.imshow(label, origin='lower', extent=[0, xmax, 0, ymax], cmap=cmap, vmin=0, vmax=3, interpolation='nearest')

RADIUS = 1.0
for (x, y), t in zip(dedup_pts, dedup_types):
    color = 'white' if t == 1 else 'pink'
    ax.add_patch(Circle((x, y), RADIUS, facecolor=color, edgecolor='black', linewidth=0.5, zorder=5))

ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout(pad=0)
plt.savefig('/workspace/ocean_eigenvector_partition.png', dpi=100)
print("Saved /workspace/ocean_eigenvector_partition.png")
