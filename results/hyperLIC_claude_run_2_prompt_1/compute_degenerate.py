import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('/workspace/brain.vti')
reader.Update()
img = reader.GetOutput()
dims = img.GetDimensions()
nx, ny, nz = dims
origin = img.GetOrigin()
spacing = img.GetSpacing()

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
D = vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

f1 = A - D      # zero where A == D
f2 = 2.0 * B    # zero where B == 0

def coord(i, j):
    x = origin[0] + i * spacing[0]
    y = origin[1] + j * spacing[1]
    return np.array([x, y, 0.0])

def bary_zero(w1, w2, w3):
    # solve l1*w1 + l2*w2 + l3*w3 = 0, l1+l2+l3 = 1
    M = np.array([
        [w1[0], w2[0], w3[0]],
        [w1[1], w2[1], w3[1]],
        [1.0,   1.0,   1.0],
    ])
    rhs = np.array([0.0, 0.0, 1.0])
    try:
        l = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    if np.all(l >= -1e-9) and np.all(l <= 1 + 1e-9):
        return l
    return None

def cross2(a, b):
    return a[0]*b[1] - a[1]*b[0]

def classify_triangle(vA, vB, vC, wA, wB, wC):
    l = bary_zero(wA, wB, wC)
    if l is None:
        return None
    l1, l2, l3 = l
    p = l1*vA + l2*vB + l3*vC
    spatial_sign = np.sign(cross2(vB[:2]-vA[:2], vC[:2]-vA[:2]))
    f_sign = np.sign(cross2(wB - wA, wC - wA))
    if spatial_sign == 0 or f_sign == 0:
        return None
    index_sign = spatial_sign * f_sign
    kind = 'wedge' if index_sign > 0 else 'trisector'
    return p, kind

points = []
kinds = []

for j in range(ny - 1):
    for i in range(nx - 1):
        v00 = coord(i, j)
        v10 = coord(i+1, j)
        v11 = coord(i+1, j+1)
        v01 = coord(i, j+1)

        w00 = np.array([f1[j, i], f2[j, i]])
        w10 = np.array([f1[j, i+1], f2[j, i+1]])
        w11 = np.array([f1[j+1, i+1], f2[j+1, i+1]])
        w01 = np.array([f1[j+1, i], f2[j+1, i]])

        for (va, vb, vc, wa, wb, wc) in [
            (v00, v10, v11, w00, w10, w11),
            (v00, v11, v01, w00, w11, w01),
        ]:
            res = classify_triangle(va, vb, vc, wa, wb, wc)
            if res is not None:
                p, kind = res
                points.append(p)
                kinds.append(kind)

print('degenerate points found:', len(points))
ntri = sum(1 for k in kinds if k == 'trisector')
nwed = sum(1 for k in kinds if k == 'wedge')
print('trisectors:', ntri, 'wedges:', nwed)

vtk_points = vtk.vtkPoints()
type_arr = vtk.vtkIntArray()
type_arr.SetName('type')  # 0 = trisector, 1 = wedge

for p, k in zip(points, kinds):
    vtk_points.InsertNextPoint(p[0], p[1], p[2])
    type_arr.InsertNextValue(0 if k == 'trisector' else 1)

poly = vtk.vtkPolyData()
poly.SetPoints(vtk_points)
poly.GetPointData().AddArray(type_arr)

verts = vtk.vtkCellArray()
for idx in range(vtk_points.GetNumberOfPoints()):
    verts.InsertNextCell(1)
    verts.InsertCellPoint(idx)
poly.SetVerts(verts)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('/workspace/degenerate_points.vtp')
writer.SetInputData(poly)
writer.Write()
print('wrote /workspace/degenerate_points.vtp')
