"""
Extract degenerate points (wedges and trisectors) from a symmetric
2x2 piecewise-linear tensor field T = [[A, B], [B, D]] defined on the
point data of brain.vti.

Classic Delmarcelle-Hesselink algorithm:
  a = (A - D) / 2 ,  b = B
Degenerate points are locations where the deviator (a, b) vanishes.
Since a, b are piecewise-linear (affine per triangle), each triangle
of the mesh contains at most one degenerate point, found by solving
the 2x2 linear system a(x,y) = 0, b(x,y) = 0.

Classification uses the sign of
  delta = da/dx * db/dy - da/dy * db/dx
  delta > 0 -> wedge
  delta < 0 -> trisector
"""
import vtk
import numpy as np

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('/workspace/brain.vti')
reader.Update()
data = reader.GetOutput()

A = data.GetPointData().GetArray('A')
B = data.GetPointData().GetArray('B')
D = data.GetPointData().GetArray('D')

n = data.GetNumberOfPoints()
a_vals = np.array([A.GetValue(i) for i in range(n)])
b_vals = np.array([B.GetValue(i) for i in range(n)])
d_vals = np.array([D.GetValue(i) for i in range(n)])

dev_a = 0.5 * (a_vals - d_vals)
dev_b = b_vals

points_out = vtk.vtkPoints()
types_out = []  # 0 = trisector, 1 = wedge


def solve_triangle(pids, pts):
    (x0, y0, _), (x1, y1, _), (x2, y2, _) = pts
    a0, a1, a2 = dev_a[pids[0]], dev_a[pids[1]], dev_a[pids[2]]
    b0, b1, b2 = dev_b[pids[0]], dev_b[pids[1]], dev_b[pids[2]]

    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x0, y2 - y0
    det = dx1 * dy2 - dx2 * dy1
    if abs(det) < 1e-14:
        return

    da1, da2 = a1 - a0, a2 - a0
    db1, db2 = b1 - b0, b2 - b0

    ax = (da1 * dy2 - da2 * dy1) / det
    ay = (dx1 * da2 - dx2 * da1) / det
    bx = (db1 * dy2 - db2 * dy1) / det
    by = (dx1 * db2 - dx2 * db1) / det

    det2 = ax * by - ay * bx
    if abs(det2) < 1e-14:
        return

    u = (-a0 * by + b0 * ay) / det2
    v = (-ax * b0 + bx * a0) / det2

    # barycentric test (same matrix as dx/dy system)
    L1 = (u * dy2 - v * dx2) / det
    L2 = (dx1 * v - dy1 * u) / det
    L0 = 1.0 - L1 - L2
    eps = -1e-9
    if L0 < eps or L1 < eps or L2 < eps:
        return

    x, y = x0 + u, y0 + v
    points_out.InsertNextPoint(x, y, 0.0)
    types_out.append(1 if det2 > 0 else 0)  # wedge : trisector


ncells = data.GetNumberOfCells()
for c in range(ncells):
    cell = data.GetCell(c)
    if cell.GetNumberOfPoints() != 4:
        continue
    pids = [cell.GetPointId(k) for k in range(4)]
    pts = [cell.GetPoints().GetPoint(k) for k in range(4)]
    # VTK_PIXEL ordering: 0=(x,y) 1=(x+1,y) 2=(x,y+1) 3=(x+1,y+1)
    # diagonal split along 0-3
    solve_triangle([pids[0], pids[1], pids[3]], [pts[0], pts[1], pts[3]])
    solve_triangle([pids[0], pids[3], pids[2]], [pts[0], pts[3], pts[2]])

poly = vtk.vtkPolyData()
poly.SetPoints(points_out)

type_arr = vtk.vtkIntArray()
type_arr.SetName('type')
for t in types_out:
    type_arr.InsertNextValue(t)
poly.GetPointData().AddArray(type_arr)

verts = vtk.vtkCellArray()
for i in range(points_out.GetNumberOfPoints()):
    verts.InsertNextCell(1)
    verts.InsertCellPoint(i)
poly.SetVerts(verts)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('/workspace/degenerate_points.vtp')
writer.SetInputData(poly)
writer.Write()

ntri = sum(1 for t in types_out if t == 0)
nwed = sum(1 for t in types_out if t == 1)
print(f'Found {len(types_out)} degenerate points: {ntri} trisectors, {nwed} wedges')
