import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("./Ocean.vti")
reader.Update()
img = reader.GetOutput()
dims = img.GetDimensions()
nx, ny = dims[0], dims[1]
spacing = img.GetSpacing()
origin = img.GetOrigin()

pd = img.GetPointData()
def arr(name):
    a = vtk_to_numpy(pd.GetArray(name)).astype(np.float64)
    return a.reshape(ny, nx)  # vtk image data is ordered x fastest

A = arr("A"); B = arr("B"); C = arr("C"); D = arr("D")

# deviator components of the asymmetric tensor (Delmarcelle-Hesselink / Zheng-Pang decomposition)
E = 0.5*(A - D)
F = 0.5*(B + C)
G = 0.5*(B - C)
Delta = E**2 + F**2 - G**2   # >0: real distinct eigenvalues, <0: complex-conjugate eigenvalues

x0, y0 = origin[0], origin[1]
sx, sy = spacing[0], spacing[1]

def coord(i, j):
    return x0 + i*sx, y0 + j*sy

# ---- locate degenerate points (common zeros of E and F) per triangulated cell ----
degenerate = []  # (x, y, kind)  kind: 'wedge' or 'trisector'

for j in range(ny-1):
    for i in range(nx-1):
        # 4 grid corners of the cell
        corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
        E_c = [E[cj, ci] for ci, cj in corners]
        F_c = [F[cj, ci] for ci, cj in corners]
        P_c = [coord(ci, cj) for ci, cj in corners]

        # split quad into 2 triangles: (0,1,2) and (0,2,3)
        tris = [(0,1,2), (0,2,3)]
        for (a,b,c) in tris:
            idxs = [a,b,c]
            pts = [P_c[k] for k in idxs]
            Ev = [E_c[k] for k in idxs]
            Fv = [F_c[k] for k in idxs]

            # linear interpolation: E(x,y)=e0+ex*x+ey*y  solved via barycentric / plane fit
            (x1,y1),(x2,y2),(x3,y3) = pts
            M = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1]])
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
            ce = Minv.dot(np.array(Ev))   # [ex, ey, e0]
            cf = Minv.dot(np.array(Fv))   # [fx, fy, f0]
            ex, ey, e0 = ce
            fx, fy, f0 = cf

            Jdet = ex*fy - ey*fx
            Amat = np.array([[ex, ey],[fx, fy]])
            if abs(np.linalg.det(Amat)) < 1e-14:
                continue
            sol = np.linalg.solve(Amat, np.array([-e0, -f0]))
            xz, yz = sol

            # barycentric check that zero lies inside this triangle
            denom = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
            if abs(denom) < 1e-14:
                continue
            l2 = ((xz-x1)*(y3-y1) - (x3-x1)*(yz-y1)) / denom
            l3 = ((x2-x1)*(yz-y1) - (xz-x1)*(y2-y1)) / denom
            l1 = 1 - l2 - l3
            eps = -1e-9
            if l1 >= eps and l2 >= eps and l3 >= eps:
                kind = "wedge" if Jdet > 0 else "trisector"
                degenerate.append((xz, yz, kind))

print("Degenerate points found:", len(degenerate))
n_w = sum(1 for p in degenerate if p[2]=="wedge")
n_t = sum(1 for p in degenerate if p[2]=="trisector")
print("wedges:", n_w, "trisectors:", n_t)

# ---- render: eigenvector partition at 10 pixels per grid square ----
ppc = 10
W = (nx-1)*ppc
H = (ny-1)*ppc

# upsample Delta field with bilinear interpolation to pixel resolution for the partition image
from scipy.ndimage import zoom
zoom_y = H/ny
zoom_x = W/nx
Delta_img = zoom(Delta, (zoom_y, zoom_x), order=1)

partition = (Delta_img >= 0).astype(np.uint8)  # 1: real distinct eigenvalues, 0: complex-conjugate

fig, ax = plt.subplots(figsize=(10,10), dpi=100)
extent = [x0, x0+(nx-1)*sx, y0, y0+(ny-1)*sy]
cmap = matplotlib.colors.ListedColormap(["#4c72b0", "#dd8452"])  # complex vs real regions
ax.imshow(partition, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")

radius = 1.0
for (xz, yz, kind) in degenerate:
    color = "pink" if kind == "trisector" else "white"
    circ = Circle((xz, yz), radius=radius, facecolor=color, edgecolor="black", linewidth=0.5, zorder=5)
    ax.add_patch(circ)

ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_aspect("equal")
ax.set_title(f"Eigenvector partition of Ocean.vti tensor field\n({n_w} wedges, {n_t} trisectors)")
plt.tight_layout()
plt.savefig("/workspace/Ocean_eigenvector_partition.png", dpi=100)
print("saved /workspace/Ocean_eigenvector_partition.png")
