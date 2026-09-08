import vtk
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from vtk.util.numpy_support import vtk_to_numpy

# ---------- 1. Read data ----------
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("/workspace/Ocean.vti")
reader.Update()
img = reader.GetOutput()

dims = img.GetDimensions()      # (nx, ny, nz)
spacing = img.GetSpacing()
origin = img.GetOrigin()
nx, ny, nz = dims
print(f"Dimensions (nx,ny,nz) = {dims}")
print(f"Spacing = {spacing}")
print(f"Origin = {origin}")

pd = img.GetPointData()
def get_field(name):
    arr = vtk_to_numpy(pd.GetArray(name)).astype(np.float64)
    # VTK point ordering is x-fastest, then y, then z. reshape to (ny,nx) [drop z=1]
    return arr.reshape(ny, nx)

A = get_field("A")
B = get_field("B")
C = get_field("C")
D = get_field("D")

# ---------- 2. Derived scalar fields ----------
a = 0.5 * (A - D)
b = 0.5 * (B + C)
c = 0.5 * (B - C)
delta = a**2 + b**2 - c**2

n_real = np.sum(delta > 0)
n_complex = np.sum(delta < 0)
print(f"Vertices with real eigenvalues (delta>0): {n_real}")
print(f"Vertices with complex eigenvalues (delta<0): {n_complex}")

# ---------- 3. Degenerate point detection via Poincare index ----------
dx, dy = spacing[0], spacing[1]
ox, oy = origin[0], origin[1]

wedge_points = []
trisector_points = []
skipped = []

for j in range(ny - 1):
    for i in range(nx - 1):
        # corners CCW: (i,j),(i+1,j),(i+1,j+1),(i,j+1)
        idxs = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
        avec = [a[jj, ii] for jj, ii in idxs]
        bvec = [b[jj, ii] for jj, ii in idxs]

        # quick reject: if all a>0 or all a<0 (and similarly no sign change possible)
        # skip only if clearly no zero can be inside (bounding box test)
        if (min(avec) > 0 or max(avec) < 0) and True:
            # a doesn't change sign -> a can't be zero anywhere in cell (bilinear a is
            # affine-combination bounded by corner values) -> no zero of (a,b)
            continue
        if (min(bvec) > 0 or max(bvec) < 0):
            continue

        angles = [np.arctan2(bvec[k], avec[k]) for k in range(4)]
        total = 0.0
        degenerate_cell = False
        for k in range(4):
            a1, b1 = avec[k], bvec[k]
            a2, b2 = avec[(k + 1) % 4], bvec[(k + 1) % 4]
            if (a1 == 0.0 and b1 == 0.0) or (a2 == 0.0 and b2 == 0.0):
                degenerate_cell = True
                break
            cross_k = a1 * b2 - a2 * b1
            dot_k = a1 * a2 + b1 * b2
            dtheta = np.arctan2(cross_k, dot_k)
            total += dtheta

        if degenerate_cell:
            skipped.append((i, j, "corner exactly zero"))
            continue

        index = total / (2.0 * np.pi)

        if abs(index) < 0.25:
            continue  # no zero enclosed

        rounded = np.round(index * 2.0) / 2.0  # nearest half-integer

        # world coords of cell center
        xw = ox + (i + 0.5) * dx
        yw = oy + (j + 0.5) * dy

        if abs(rounded - 0.5) < 1e-6:
            wedge_points.append((xw, yw, i, j, index))
        elif abs(rounded + 0.5) < 1e-6:
            trisector_points.append((xw, yw, i, j, index))
        elif abs(rounded - 1.0) < 1e-6:
            # two coincident/unresolved wedge points at this grid resolution
            wedge_points.append((xw, yw, i, j, index))
            wedge_points.append((xw, yw, i, j, index))
        elif abs(rounded + 1.0) < 1e-6:
            # two coincident/unresolved trisector points at this grid resolution
            trisector_points.append((xw, yw, i, j, index))
            trisector_points.append((xw, yw, i, j, index))
        else:
            skipped.append((i, j, f"index={index:.4f}"))

print(f"Wedge points found: {len(wedge_points)}")
print(f"Trisector points found: {len(trisector_points)}")
print(f"Skipped/ambiguous cells: {len(skipped)}")
if skipped:
    print("  examples:", skipped[:5])

# ---------- 4. Visualization ----------
ncell_x = nx - 1
ncell_y = ny - 1
px_per_cell = 10
raster_w = ncell_x * px_per_cell
raster_h = ncell_y * px_per_cell

# delta at cell centers via averaging the 4 corners (bilinear-consistent-ish),
# then piecewise-constant upsample per cell for a clean partition look
delta_cell = 0.25 * (
    delta[0:ny - 1, 0:nx - 1] + delta[0:ny - 1, 1:nx] +
    delta[1:ny, 0:nx - 1] + delta[1:ny, 1:nx]
)
sign_cell = np.where(delta_cell > 0, 1.0, -1.0)

raster = np.repeat(np.repeat(sign_cell, px_per_cell, axis=0), px_per_cell, axis=1)

dpi = 100
figsize = (raster_w / dpi, raster_h / dpi)
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

# extent in "grid index" coordinates (0..ncell_x, 0..ncell_y), matching
# radius-1 = one grid-cell-unit circles naturally
cmap = matplotlib.colors.ListedColormap(["#3B6FA0", "#E8A33D"])  # complex(-1)=blue, real(+1)=orange
norm = matplotlib.colors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)

ax.imshow(raster, cmap=cmap, norm=norm, origin="lower",
          extent=[0, ncell_x, 0, ncell_y], interpolation="nearest")

radius = 1.0
for (xw, yw, i, j, idx) in trisector_points:
    ax.add_patch(Circle((i + 0.5, j + 0.5), radius=radius,
                         facecolor="pink", edgecolor="black", linewidth=0.5, zorder=3))
for (xw, yw, i, j, idx) in wedge_points:
    ax.add_patch(Circle((i + 0.5, j + 0.5), radius=radius,
                         facecolor="white", edgecolor="black", linewidth=0.5, zorder=3))

ax.set_xlim(0, ncell_x)
ax.set_ylim(0, ncell_y)
ax.set_aspect("equal")
ax.set_title("Ocean tensor field topology: eigenvector partition + degenerate points\n"
             "(orange = real eigenvalues, blue = complex eigenvalues; "
             "white = wedge, pink = trisector)", fontsize=8)
ax.set_xlabel("grid cell index (x)")
ax.set_ylabel("grid cell index (y)")

legend_handles = [
    matplotlib.patches.Patch(facecolor="#E8A33D", label="Real eigenvalues (delta>0)"),
    matplotlib.patches.Patch(facecolor="#3B6FA0", label="Complex eigenvalues (delta<0)"),
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                             markeredgecolor='black', markersize=8, label='Wedge point'),
    matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink',
                             markeredgecolor='black', markersize=8, label='Trisector point'),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=6, framealpha=0.8)

plt.tight_layout()
plt.savefig("/workspace/ocean_tensor_topology.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)

# ---------- 5. Save CSV + summary ----------
with open("/workspace/ocean_degenerate_points.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type", "x_world", "y_world", "cell_i", "cell_j", "poincare_index"])
    for (xw, yw, i, j, idx) in wedge_points:
        w.writerow(["wedge", xw, yw, i, j, idx])
    for (xw, yw, i, j, idx) in trisector_points:
        w.writerow(["trisector", xw, yw, i, j, idx])

print("\n===== SUMMARY =====")
print(f"Grid dims (nx,ny): ({nx},{ny})   cells: ({ncell_x},{ncell_y})")
print(f"Spacing: {spacing}   Origin: {origin}")
print(f"Wedge points: {len(wedge_points)}")
print(f"Trisector points: {len(trisector_points)}")
print(f"Total degenerate points: {len(wedge_points) + len(trisector_points)}")
print(f"Total cells: {ncell_x * ncell_y}")
print(f"Skipped/ambiguous cells: {len(skipped)}")
print("Wrote /workspace/ocean_tensor_topology.png")
print("Wrote /workspace/ocean_degenerate_points.csv")
