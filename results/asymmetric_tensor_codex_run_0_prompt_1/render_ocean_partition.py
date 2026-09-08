#!/usr/bin/env pvpython
"""Render the eigenvector partition of the 2-D asymmetric tensor Ocean.vti."""
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "Ocean.vti")
OUTPUT = os.path.join(HERE, "Ocean_eigenvector_partition.png")
POINTS = os.path.join(HERE, "Ocean_degenerate_points.csv")
SAMPLES_PER_CELL = 10

# The four regions of the eigenvector manifold, chosen for clear contrast.
# (real/complex) x (counterclockwise/clockwise rotation)
COLORS = np.array([
    ( 74, 144, 226),  # real, counterclockwise
    (245, 166,  35),  # complex, counterclockwise
    ( 92,  72, 160),  # real, clockwise
    ( 35, 174, 154),  # complex, clockwise
], dtype=np.uint8)

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INPUT)
reader.Update()
data = reader.GetOutput()
nx, ny, nz = data.GetDimensions()
if nz != 1:
    raise RuntimeError("This renderer expects a 2-D tensor field.")

def field(name):
    return vtk_to_numpy(data.GetPointData().GetArray(name)).reshape(ny, nx)

A, B, C, D = (field(name) for name in ("A", "B", "C", "D"))
# T = gamma_d I + [[gamma_a, gamma_b], [gamma_b, -gamma_a]]
#                  + [[0,-gamma_r], [gamma_r,0]]
ga = (A - D) * 0.5
gb = (B + C) * 0.5
gr = (C - B) * 0.5
gs = np.hypot(ga, gb)

# Subdivide each original square consistently along lower-left to upper-right.
# This makes the component-wise interpolation explicitly piecewise linear.
npx, npy = (nx - 1) * SAMPLES_PER_CELL, (ny - 1) * SAMPLES_PER_CELL
xx = (np.arange(npx) + .5) / SAMPLES_PER_CELL
yy = (np.arange(npy) + .5) / SAMPLES_PER_CELL
X, Y = np.meshgrid(xx, yy)
i = np.minimum(X.astype(int), nx - 2)
j = np.minimum(Y.astype(int), ny - 2)
u, v = X - i, Y - j

def pl(f):
    f00, f10 = f[j, i], f[j, i + 1]
    f01, f11 = f[j + 1, i], f[j + 1, i + 1]
    # Triangles (00,10,11) for u >= v and (00,11,01) otherwise.
    return np.where(u >= v,
                    f00 + u * (f10 - f00) + v * (f11 - f10),
                    f00 + u * (f11 - f01) + v * (f01 - f00))

GRA, GRB, GR = pl(ga), pl(gb), pl(gr)
GS = np.hypot(GRA, GRB)
# 0: real+CCW, 1: complex+CCW, 2: real+CW, 3: complex+CW
classes = np.where(GR >= 0, np.where(np.abs(GR) <= GS, 0, 1),
                   np.where(np.abs(GR) <= GS, 2, 3))
rgb = COLORS[classes]

# Degenerate points are zeros of the symmetric anisotropic component
# (gamma_a,gamma_b).  A zero of a PL vector field is found once per triangle.
degenerate = []
seen = set()
for j0 in range(ny - 1):
    for i0 in range(nx - 1):
        corners = ((i0, j0), (i0 + 1, j0), (i0 + 1, j0 + 1), (i0, j0 + 1))
        for tri in ((corners[0], corners[1], corners[2]),
                    (corners[0], corners[2], corners[3])):
            p = np.array([[ga[y, x], gb[y, x]] for x, y in tri])
            M = np.column_stack((p[1] - p[0], p[2] - p[0]))
            det = np.linalg.det(M)
            if abs(det) < 1e-12:
                continue
            bary = np.linalg.solve(M, -p[0])
            if min(bary[0], bary[1]) >= -1e-9 and bary.sum() <= 1 + 1e-9:
                x = tri[0][0] + bary[0] * (tri[1][0] - tri[0][0]) + bary[1] * (tri[2][0] - tri[0][0])
                y = tri[0][1] + bary[0] * (tri[1][1] - tri[0][1]) + bary[1] * (tri[2][1] - tri[0][1])
                key = (round(x, 8), round(y, 8))
                if key not in seen:
                    seen.add(key)
                    # Positive line-field index is a wedge; negative is a trisector.
                    degenerate.append((x, y, "wedge" if det > 0 else "trisector"))

with open(POINTS, "w", encoding="utf-8") as f:
    f.write("x,y,type\n")
    for x, y, kind in degenerate:
        f.write(f"{x:.9g},{y:.9g},{kind}\n")

# Add the requested singularity discs to the 10-pixel-per-square raster.
# (A world radius of 1 is therefore exactly 10 pixels.)
for x, y, kind in degenerate:
    cx, cy = x * SAMPLES_PER_CELL, y * SAMPLES_PER_CELL
    yy0, xx0 = np.ogrid[:npy, :npx]
    mask = (xx0 - cx) ** 2 + (yy0 - cy) ** 2 <= SAMPLES_PER_CELL ** 2
    rgb[mask] = (255, 255, 255) if kind == "wedge" else (255, 89, 166)

# Write the exact raster with VTK, the rendering library underlying ParaView.
# The image dimensions are exactly 10 pixels for each of the 100 by 100 cells.
image = vtk.vtkImageData()
image.SetDimensions(npx, npy, 1)
image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
arr = numpy_to_vtk(rgb.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
arr.SetNumberOfComponents(3)
arr.SetName("Eigenvector partition")
image.GetPointData().SetScalars(arr)
writer = vtk.vtkPNGWriter()
writer.SetFileName(OUTPUT)
writer.SetInputData(image)
writer.Write()
print(f"Wrote {OUTPUT} ({npx}x{npy}); {len(degenerate)} degenerate points")
print(f"Wrote {POINTS}")
