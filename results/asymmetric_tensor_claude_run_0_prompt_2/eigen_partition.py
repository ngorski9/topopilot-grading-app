import vtk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PX_PER_UNIT = 10
RADIUS = 1.0

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("Ocean.vti")
reader.Update()
img = reader.GetOutput()
nx, ny, _ = img.GetDimensions()
sx, sy, _ = img.GetSpacing()
ox, oy, _ = img.GetOrigin()

def get_arr(name):
    from vtk.util.numpy_support import vtk_to_numpy
    a = vtk_to_numpy(img.GetPointData().GetArray(name))
    return a.reshape(ny, nx)  # vtk point ordering: x fastest

A = get_arr("A")
B = get_arr("B")
C = get_arr("C")
D = get_arr("D")

# grid point coordinates
xs = ox + sx * np.arange(nx)
ys = oy + sy * np.arange(ny)

ncx, ncy = nx - 1, ny - 1  # number of cells

# ---- Build triangles for each cell: (v00, v10, v11) and (v00, v11, v01) ----
def cell_vertex_vals(arr, i, j):
    # i: x index 0..ncx-1, j: y index 0..ncy-1
    v00 = arr[j, i]
    v10 = arr[j, i + 1]
    v01 = arr[j + 1, i]
    v11 = arr[j + 1, i + 1]
    return v00, v10, v01, v11

def grad_lin(g0, g1, g2, x0, y0, x1, y1, x2, y2):
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    dgdx = ((g1 - g0) * (y2 - y0) - (g2 - g0) * (y1 - y0)) / area2
    dgdy = ((g2 - g0) * (x1 - x0) - (g1 - g0) * (x2 - x0)) / area2
    return dgdx, dgdy

degenerate_points = []  # (x, y, kind)  kind: 'wedge' or 'trisector'

I, J = np.meshgrid(np.arange(ncx), np.arange(ncy))
I = I.ravel()
J = J.ravel()

X0 = xs[I]
Y0 = ys[J]
X1 = xs[I + 1]
Y1 = ys[J]
X2 = xs[I]
Y2 = ys[J + 1]
X3 = xs[I + 1]
Y3 = ys[J + 1]

A00 = A[J, I]; A10 = A[J, I + 1]; A01 = A[J + 1, I]; A11 = A[J + 1, I + 1]
B00 = B[J, I]; B10 = B[J, I + 1]; B01 = B[J + 1, I]; B11 = B[J + 1, I + 1]
C00 = C[J, I]; C10 = C[J, I + 1]; C01 = C[J + 1, I]; C11 = C[J + 1, I + 1]
D00 = D[J, I]; D10 = D[J, I + 1]; D01 = D[J + 1, I]; D11 = D[J + 1, I + 1]

E00 = A00 - D00; E10 = A10 - D10; E01 = A01 - D01; E11 = A11 - D11
F00 = B00 + C00; F10 = B10 + C10; F01 = B01 + C01; F11 = B11 + C11

def solve_triangle(p0x, p0y, p1x, p1y, p2x, p2y, e0, e1, e2, f0, f1, f2):
    # solve l1*(e1-e0)+l2*(e2-e0) = -e0 ; l1*(f1-f0)+l2*(f2-f0) = -f0
    a11 = e1 - e0
    a12 = e2 - e0
    a21 = f1 - f0
    a22 = f2 - f0
    det = a11 * a22 - a12 * a21
    valid = np.abs(det) > 1e-14
    l1 = np.zeros_like(det)
    l2 = np.zeros_like(det)
    l1[valid] = (-e0[valid] * a22[valid] + a12[valid] * f0[valid]) / det[valid]
    l2[valid] = (-f0[valid] * a11[valid] + a21[valid] * e0[valid]) / det[valid]
    l0 = 1.0 - l1 - l2
    inside = valid & (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
    px = p0x + l1 * (p1x - p0x) + l2 * (p2x - p0x)
    py = p0y + l1 * (p1y - p0y) + l2 * (p2y - p0y)
    dEdx, dEdy = grad_lin(e0, e1, e2, p0x, p0y, p1x, p1y, p2x, p2y)
    dFdx, dFdy = grad_lin(f0, f1, f2, p0x, p0y, p1x, p1y, p2x, p2y)
    jac = dEdx * dFdy - dEdy * dFdx
    return inside, px, py, jac

# Triangle A: v00, v10, v11
insideA, pxA, pyA, jacA = solve_triangle(
    X0, Y0, X1, Y1, X3, Y3,
    E00, E10, E11, F00, F10, F11)

# Triangle B: v00, v11, v01
insideB, pxB, pyB, jacB = solve_triangle(
    X0, Y0, X3, Y3, X2, Y2,
    E00, E11, E01, F00, F11, F01)

for mask, px, py, jac in ((insideA, pxA, pyA, jacA), (insideB, pxB, pyB, jacB)):
    idxs = np.where(mask)[0]
    for k in idxs:
        kind = "wedge" if jac[k] > 0 else "trisector"
        degenerate_points.append((px[k], py[k], kind))

print("Found", len(degenerate_points), "degenerate points")
n_wedge = sum(1 for p in degenerate_points if p[2] == "wedge")
n_tri = sum(1 for p in degenerate_points if p[2] == "trisector")
print("wedges:", n_wedge, "trisectors:", n_tri)

# ---- Render eigenvector partition ----
# Partition by sign of discriminant Delta = (A-D)^2 + 4*B*C  (real vs complex eigenvalues)
# Evaluated per-triangle via linear interpolation (barycentric on a fine pixel grid).

width_px = ncx * PX_PER_UNIT
height_px = ncy * PX_PER_UNIT

partition = np.zeros((height_px, width_px), dtype=np.uint8)

# For each cell, rasterize its 10x10 pixel block using linear interpolation on the two triangles
sub = np.linspace(0, 1, PX_PER_UNIT, endpoint=False) + 0.5 / PX_PER_UNIT
su, sv = np.meshgrid(sub, sub)  # local coords within cell, su: x-frac, sv: y-frac

for idx in range(len(I)):
    i, j = I[idx], J[idx]
    a00, a10, a01, a11 = A00[idx], A10[idx], A01[idx], A11[idx]
    b00, b10, b01, b11 = B00[idx], B10[idx], B01[idx], B11[idx]
    c00, c10, c01, c11 = C00[idx], C10[idx], C01[idx], C11[idx]
    d00, d10, d01, d11 = D00[idx], D10[idx], D01[idx], D11[idx]

    # split into two triangles based on su+sv <= 1 (v00,v10,v01 triangle... ) but we used diagonal v00-v11
    # local coords: triangle A = {su>=sv? } Actually diagonal from (0,0)-(1,1): triangle A (v00,v10,v11) is where su>=sv... let's use su>sv -> lower-right triangle (v00,v10,v11); su<=sv -> upper-left triangle (v00,v11,v01)
    maskA = su >= sv
    maskB = ~maskA

    def bilinear_tri(v00, v10, v01, v11, maskA, maskB, su, sv):
        out = np.zeros_like(su)
        # triangle A: v00,v10,v11 ; barycentric via (su,sv): l0=1-su, l1=su-sv, l2=sv  (since along su=sv is the diagonal)
        l0a = 1 - su[maskA]
        l1a = su[maskA] - sv[maskA]
        l2a = sv[maskA]
        out[maskA] = l0a * v00 + l1a * v10 + l2a * v11
        # triangle B: v00,v11,v01 ; l0=1-sv, l1=sv-su, l2=su
        l0b = 1 - sv[maskB]
        l1b = sv[maskB] - su[maskB]
        l2b = su[maskB]
        out[maskB] = l0b * v00 + l1b * v11 + l2b * v01
        return out

    a_ = bilinear_tri(a00, a10, a01, a11, maskA, maskB, su, sv)
    b_ = bilinear_tri(b00, b10, b01, b11, maskA, maskB, su, sv)
    c_ = bilinear_tri(c00, c10, c01, c11, maskA, maskB, su, sv)
    d_ = bilinear_tri(d00, d10, d01, d11, maskA, maskB, su, sv)

    delta = (a_ - d_) ** 2 + 4.0 * b_ * c_
    block = (delta >= 0).astype(np.uint8)

    y0 = j * PX_PER_UNIT
    x0 = i * PX_PER_UNIT
    partition[y0:y0 + PX_PER_UNIT, x0:x0 + PX_PER_UNIT] = block

# ---- Plot ----
fig_w = width_px / 100.0
fig_h = height_px / 100.0
fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])

cmap = matplotlib.colors.ListedColormap(["#3b6fb5", "#e8d98a"])  # complex-eigenvalue region, real-eigenvalue region
ax.imshow(partition, origin="lower", cmap=cmap, extent=[xs[0], xs[-1], ys[0], ys[-1]], interpolation="nearest")

for (px, py, kind) in degenerate_points:
    color = "pink" if kind == "trisector" else "white"
    circ = Circle((px, py), radius=RADIUS, facecolor=color, edgecolor="black", linewidth=0.5, zorder=5)
    ax.add_patch(circ)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])
ax.set_aspect("equal")
ax.axis("off")

plt.savefig("Ocean_eigenvector_partition.png", dpi=100)
print("Saved Ocean_eigenvector_partition.png")
