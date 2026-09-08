"""Render the 2-D tensor eigendirection partition and its PL singularities."""
import numpy as np
from scipy.optimize import root
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

INPUT = "Ocean.vti"
OUTPUT = "Ocean_eigenvector_partition.png"
POINTS = "Ocean_degenerate_points.csv"
PIXELS_PER_CELL = 10

reader = vtkXMLImageDataReader()
reader.SetFileName(INPUT)
reader.Update()
image = reader.GetOutput()
dims = image.GetDimensions()
nx, ny = dims[0], dims[1]

def array(name):
    # VTK ImageData stores x as the fastest-varying coordinate.
    return vtk_to_numpy(image.GetPointData().GetArray(name)).reshape(ny, nx)

A, B, C, D = (array(name) for name in ("A", "B", "C", "D"))

# The principal eigendirection of the planar tensor's symmetric component.
# Its two independent anisotropy components vanish exactly at tensor
# degeneracies, while the skew-symmetric component has no eigendirection.
u = A - D
v = B + C

# Sample the piecewise-bilinear field at the requested raster resolution.
width, height = (nx - 1) * PIXELS_PER_CELL, (ny - 1) * PIXELS_PER_CELL
xs = (np.arange(width) + 0.5) / PIXELS_PER_CELL
ys = (np.arange(height) + 0.5) / PIXELS_PER_CELL
ix = np.minimum(xs.astype(int), nx - 2)
iy = np.minimum(ys.astype(int), ny - 2)
fx = xs - ix
fy = ys - iy

def bilinear(values):
    q00 = values[iy[:, None], ix[None, :]]
    q10 = values[iy[:, None], ix[None, :] + 1]
    q01 = values[iy[:, None] + 1, ix[None, :]]
    q11 = values[iy[:, None] + 1, ix[None, :] + 1]
    return ((1-fy[:, None]) * ((1-fx[None, :])*q00 + fx[None, :]*q10)
            + fy[:, None] * ((1-fx[None, :])*q01 + fx[None, :]*q11))

U, V = bilinear(u), bilinear(v)
theta = 0.5 * np.arctan2(V, U)  # line direction, in [-pi/2, pi/2]
# A cyclic, high-contrast direction partition: antipodal directions match.
hue = (theta + np.pi/2) / np.pi
rgb = hsv_to_rgb(np.dstack((hue, np.full_like(hue, .72), np.full_like(hue, .96))))

def cell_value(field, j, i, x, y):
    return ((1-x)*(1-y)*field[j, i] + x*(1-y)*field[j, i+1]
            + (1-x)*y*field[j+1, i] + x*y*field[j+1, i+1])

singularities = []
for j in range(ny - 1):
    for i in range(nx - 1):
        uc = u[j:j+2, i:i+2]
        vc = v[j:j+2, i:i+2]
        if not (uc.min() <= 0 <= uc.max() and vc.min() <= 0 <= vc.max()):
            continue
        def fun(p):
            return [cell_value(u, j, i, p[0], p[1]), cell_value(v, j, i, p[0], p[1])]
        sol = root(fun, (.5, .5))
        if not sol.success or np.any(sol.x < -1e-8) or np.any(sol.x > 1+1e-8):
            continue
        x, y = np.clip(sol.x, 0, 1)
        if np.linalg.norm(fun((x, y))) > 1e-8:
            continue
        # Index sign is the local winding of (u,v); +1/2 is a wedge and
        # -1/2 is a trisector for the resulting eigenvector line field.
        du_dx = (1-y)*(u[j,i+1]-u[j,i]) + y*(u[j+1,i+1]-u[j+1,i])
        du_dy = (1-x)*(u[j+1,i]-u[j,i]) + x*(u[j+1,i+1]-u[j,i+1])
        dv_dx = (1-y)*(v[j,i+1]-v[j,i]) + y*(v[j+1,i+1]-v[j+1,i])
        dv_dy = (1-x)*(v[j+1,i]-v[j,i]) + x*(v[j+1,i+1]-v[j,i+1])
        det = du_dx*dv_dy - du_dy*dv_dx
        kind = "wedge" if det > 0 else "trisector"
        candidate = (i+x, j+y, kind, det)
        # A zero on a cell edge is encountered by both adjacent cells.
        if not any((candidate[0]-q[0])**2 + (candidate[1]-q[1])**2 < 1e-12
                   for q in singularities):
            singularities.append(candidate)

# Paint radius-one data-unit singularity disks directly in the raster.  This
# preserves the exact requested radius independently of PNG display scaling.
yy, xx = np.ogrid[:height, :width]
for x, y, kind, det in singularities:
    mask = (xx / PIXELS_PER_CELL - x)**2 + (yy / PIXELS_PER_CELL - y)**2 <= 1.0
    rgb[mask] = (1.0, 1.0, 1.0) if kind == "wedge" else (1.0, 0.41, 0.71)

plt.imsave(OUTPUT, rgb, origin="lower")
with open(POINTS, "w", encoding="utf-8") as f:
    f.write("x,y,type,jacobian_determinant,radius\n")
    for x, y, kind, det in singularities:
        f.write(f"{x:.9f},{y:.9f},{kind},{det:.12g},1\n")
print(f"wrote {OUTPUT}: {width}x{height} pixels")
print(f"wrote {POINTS}: {len(singularities)} points; "
      f"{sum(k == 'wedge' for _, _, k, _ in singularities)} wedges, "
      f"{sum(k == 'trisector' for _, _, k, _ in singularities)} trisectors")
