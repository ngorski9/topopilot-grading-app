"""
Compute degenerate points (wedges and trisectors) of a 2D symmetric
piecewise-linear tensor field T = [[A, B], [B, D]].

Standard (Delmarcelle-Hesselink 1994 / Tricoche et al. 2001) approach:
  deviator components  e = (A - D) / 2 ,  f = B
  degenerate points  <=>  e = f = 0
  classification via the sign of the local Jacobian of (e, f):
      delta > 0  ->  wedge      (Poincare index +1/2)
      delta < 0  ->  trisector  (Poincare index -1/2)

Each quad cell of the image is split into 2 triangles; (e, f) is linear
on each triangle, so we solve for the barycentric weights that zero it
out and keep the solution if it falls inside the triangle.
"""
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('/workspace/brain.vti')
reader.Update()
img = reader.GetOutput()

dims = img.GetDimensions()
nx, ny, nz = dims
origin = np.array(img.GetOrigin())
spacing = np.array(img.GetSpacing())

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
D = vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

E = 0.5 * (A - D)
F = B

def xy(i, j):
    return np.array([origin[0] + i * spacing[0], origin[1] + j * spacing[1], 0.0])

points = []
types = []  # 0 = trisector, 1 = wedge

for j in range(ny - 1):
    for i in range(nx - 1):
        # vertices of the quad cell, CCW order
        idx = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        e_v = [E[jj, ii] for ii, jj in idx]
        f_v = [F[jj, ii] for ii, jj in idx]
        p_v = [xy(ii, jj) for ii, jj in idx]

        # split quad -> 2 triangles (0,1,2) and (0,2,3), both CCW
        tris = [(0, 1, 2), (0, 2, 3)]
        for (a, b, c) in tris:
            e0, e1, e2 = e_v[a], e_v[b], e_v[c]
            f0, f1, f2 = f_v[a], f_v[b], f_v[c]
            p0, p1, p2 = p_v[a], p_v[b], p_v[c]

            # solve barycentric weights w0+w1+w2=1, w.e=0, w.f=0
            M = np.array([[e0, e1, e2],
                          [f0, f1, f2],
                          [1.0, 1.0, 1.0]])
            rhs = np.array([0.0, 0.0, 1.0])
            det = np.linalg.det(M)
            if abs(det) < 1e-300:
                continue
            w = np.linalg.solve(M, rhs)
            if np.all(w >= -1e-9) and np.all(w <= 1 + 1e-9):
                pos = w[0] * p0 + w[1] * p1 + w[2] * p2

                # Jacobian sign of (e,f) wrt (x,y) on this (CCW, positive-area) triangle:
                # signed area in (e,f) space with the same vertex ordering
                delta = (e1 - e0) * (f2 - f0) - (f1 - f0) * (e2 - e0)

                points.append(pos)
                types.append(1 if delta > 0 else 0)  # wedge : trisector

points = np.array(points)
types = np.array(types, dtype=np.int32)

print('degenerate points found:', len(points))
print('wedges:', int((types == 1).sum()), 'trisectors:', int((types == 0).sum()))

# --- write to a vtkPolyData / .vtp for ParaView ---
vpoints = vtk.vtkPoints()
for p in points:
    vpoints.InsertNextPoint(p[0], p[1], p[2])

poly = vtk.vtkPolyData()
poly.SetPoints(vpoints)

verts = vtk.vtkCellArray()
for k in range(len(points)):
    verts.InsertNextCell(1, [k])
poly.SetVerts(verts)

type_arr = vtk.vtkIntArray()
type_arr.SetName('DegenerateType')
for t in types:
    type_arr.InsertNextValue(int(t))
poly.GetPointData().AddArray(type_arr)
poly.GetPointData().SetActiveScalars('DegenerateType')

# human readable label too
label_arr = vtk.vtkStringArray()
label_arr.SetName('DegenerateTypeName')
for t in types:
    label_arr.InsertNextValue('wedge' if t == 1 else 'trisector')
poly.GetPointData().AddArray(label_arr)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('/workspace/degenerate_points.vtp')
writer.SetInputData(poly)
writer.Write()
print('wrote /workspace/degenerate_points.vtp')
