"""
Compute degenerate points (wedges and trisectors) of a piecewise-linear
symmetric 2D tensor field T = [[A, B], [B, D]] defined on brain.vti,
and visualize the tensor field together with the degenerate points.

Algorithm (Delmarcelle & Hesselink, 1994):
  Deviator components: e = (A - D) / 2 ,  f = B
  Degenerate points occur where e = f = 0.
  The grid is triangulated (2 triangles per pixel quad); within each
  triangle e and f are linear, so there is at most one zero, found via
  barycentric interpolation. The point is classified by comparing the
  orientation of the triangle in (x,y) with its orientation in (e,f):
  matching orientation -> wedge (index +1/2); opposite -> trisector (index -1/2).
"""
import vtk
from vtk.util import numpy_support as vnp
import numpy as np

INPUT = "/workspace/brain.vti"

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INPUT)
reader.Update()
img = reader.GetOutput()

nx, ny, nz = img.GetDimensions()
assert nz == 1, "expected a 2D field"
ox, oy, oz = img.GetOrigin()
sx, sy, sz = img.GetSpacing()

A = vnp.vtk_to_numpy(img.GetPointData().GetArray("A")).reshape(ny, nx)
B = vnp.vtk_to_numpy(img.GetPointData().GetArray("B")).reshape(ny, nx)
D = vnp.vtk_to_numpy(img.GetPointData().GetArray("D")).reshape(ny, nx)

E = 0.5 * (A - D)
F = B.copy()

def coords(i, j):
    return np.array([ox + i * sx, oy + j * sy, 0.0])

def ef(i, j):
    return np.array([E[j, i], F[j, i]])

def sign(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)

def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]

def process_triangle(idx0, idx1, idx2):
    """idx* are (i,j) grid indices. Returns (point, kind) or None."""
    p0, p1, p2 = coords(*idx0), coords(*idx1), coords(*idx2)
    ef0, ef1, ef2 = ef(*idx0), ef(*idx1), ef(*idx2)

    # Solve for barycentric coords (l0,l1,l2), l0+l1+l2=1, with
    # l0*ef0 + l1*ef1 + l2*ef2 = (0,0)
    # => l1*(ef1-ef0) + l2*(ef2-ef0) = -ef0
    M = np.array([ef1 - ef0, ef2 - ef0]).T  # 2x2
    detM = np.linalg.det(M)
    if abs(detM) < 1e-14:
        return None
    rhs = -ef0
    sol = np.linalg.solve(M, rhs)
    l1, l2 = sol
    l0 = 1.0 - l1 - l2
    eps = -1e-9
    if l0 < eps or l1 < eps or l2 < eps:
        return None  # zero not inside this triangle

    point = l0 * p0 + l1 * p1 + l2 * p2

    orient_xy = sign(cross2((p1 - p0)[:2], (p2 - p0)[:2]))
    orient_ef = sign(cross2(ef1 - ef0, ef2 - ef0))
    if orient_xy == 0 or orient_ef == 0:
        return None

    kind = "wedge" if orient_xy == orient_ef else "trisector"
    return point, kind

degenerate_points = []  # (point, kind)

for j in range(ny - 1):
    for i in range(nx - 1):
        # quad corners
        c00 = (i, j)
        c10 = (i + 1, j)
        c11 = (i + 1, j + 1)
        c01 = (i, j + 1)
        # split into 2 triangles (consistent diagonal)
        for tri in ((c00, c10, c11), (c00, c11, c01)):
            res = process_triangle(*tri)
            if res is not None:
                degenerate_points.append(res)

print("Found %d degenerate points" % len(degenerate_points))
n_wedge = sum(1 for _, k in degenerate_points if k == "wedge")
n_tri = sum(1 for _, k in degenerate_points if k == "trisector")
print("  wedges:", n_wedge, " trisectors:", n_tri)

# ---- Write degenerate points to a .vtp file for use in ParaView ----
points = vtk.vtkPoints()
kind_arr = vtk.vtkIntArray()
kind_arr.SetName("DegenerateType")  # 0 = trisector, 1 = wedge
verts = vtk.vtkCellArray()

for k, (pt, kind) in enumerate(degenerate_points):
    pid = points.InsertNextPoint(pt[0], pt[1], pt[2])
    kind_arr.InsertNextValue(1 if kind == "wedge" else 0)
    verts.InsertNextCell(1)
    verts.InsertCellPoint(pid)

poly = vtk.vtkPolyData()
poly.SetPoints(points)
poly.SetVerts(verts)
poly.GetPointData().AddArray(kind_arr)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/workspace/degenerate_points.vtp")
writer.SetInputData(poly)
writer.Write()

print("Wrote /workspace/degenerate_points.vtp")
