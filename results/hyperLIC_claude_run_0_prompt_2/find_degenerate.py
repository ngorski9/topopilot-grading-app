import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("/workspace/brain.vti")
reader.Update()
img = reader.GetOutput()
dims = img.GetDimensions()
nx, ny, nz = dims
assert nz == 1

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray("A")).reshape(ny, nx)
B = vtk_to_numpy(pd.GetArray("B")).reshape(ny, nx)
D = vtk_to_numpy(pd.GetArray("D")).reshape(ny, nx)

origin = img.GetOrigin()
spacing = img.GetSpacing()

E = A - D
F = 2.0 * B

def pt(i, j):
    x = origin[0] + i * spacing[0]
    y = origin[1] + j * spacing[1]
    return np.array([x, y, 0.0])

def solve_tri(idx_list):
    # idx_list: list of (i,j) for 3 verts
    P = [pt(i, j) for (i, j) in idx_list]
    Ev = [E[j, i] for (i, j) in idx_list]
    Fv = [F[j, i] for (i, j) in idx_list]
    E1, E2, E3 = Ev
    F1, F2, F3 = Fv
    # solve l1*(E1-E3)+l2*(E2-E3) = -E3 ; l1*(F1-F3)+l2*(F2-F3) = -F3
    a11 = E1 - E3; a12 = E2 - E3; b1 = -E3
    a21 = F1 - F3; a22 = F2 - F3; b2 = -F3
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-14:
        return None
    l1 = (b1 * a22 - a12 * b2) / det
    l2 = (a11 * b2 - b1 * a21) / det
    l3 = 1.0 - l1 - l2
    eps = -1e-9
    if l1 < eps or l2 < eps or l3 < eps:
        return None
    l1 = max(l1, 0.0); l2 = max(l2, 0.0); l3 = max(l3, 0.0)
    loc = l1 * P[0] + l2 * P[1] + l3 * P[2]
    cross = (E2 - E1) * (F3 - F1) - (E3 - E1) * (F2 - F1)
    kind = "wedge" if cross > 0 else "trisector"
    return loc, kind

results = []
for j in range(ny - 1):
    for i in range(nx - 1):
        for tri in ( [(i, j), (i + 1, j), (i + 1, j + 1)],
                     [(i, j), (i + 1, j + 1), (i, j + 1)] ):
            r = solve_tri(tri)
            if r is not None:
                results.append(r)

print("Total degenerate points found:", len(results))
n_wedge = sum(1 for _, k in results if k == "wedge")
n_tri = sum(1 for _, k in results if k == "trisector")
print("Wedges:", n_wedge, "Trisectors:", n_tri)

# Build vtkPolyData
points = vtk.vtkPoints()
verts = vtk.vtkCellArray()
type_arr = vtk.vtkIntArray()
type_arr.SetName("DegenerateType")  # 0 = trisector, 1 = wedge

for loc, kind in results:
    pid = points.InsertNextPoint(loc[0], loc[1], loc[2])
    verts.InsertNextCell(1)
    verts.InsertCellPoint(pid)
    type_arr.InsertNextValue(1 if kind == "wedge" else 0)

poly = vtk.vtkPolyData()
poly.SetPoints(points)
poly.SetVerts(verts)
poly.GetPointData().AddArray(type_arr)
poly.GetPointData().SetActiveScalars("DegenerateType")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/workspace/degenerate_points.vtp")
writer.SetInputData(poly)
writer.Write()
print("Wrote /workspace/degenerate_points.vtp")
