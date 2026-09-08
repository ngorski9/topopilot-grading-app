import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("brain.vti")
reader.Update()
img = reader.GetOutput()

dims = img.GetDimensions()
nx, ny, nz = dims
origin = np.array(img.GetOrigin())
spacing = np.array(img.GetSpacing())

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray("A")).astype(np.float64)
B = vtk_to_numpy(pd.GetArray("B")).astype(np.float64)
D = vtk_to_numpy(pd.GetArray("D")).astype(np.float64)

delta = A - D
beta = 2.0 * B

def idx(i, j):
    return i + j * nx

# precompute point positions (x,y,0)
xs = origin[0] + spacing[0] * np.arange(nx)
ys = origin[1] + spacing[1] * np.arange(ny)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

points_out = []
types_out = []  # 0 = trisector, 1 = wedge

def process_triangle(ia, ja, ib, jb, ic, jc):
    i0, j0 = idx(ia, ja), None  # placeholder unused
    p0 = idx(ia, ja)
    p1 = idx(ib, jb)
    p2 = idx(ic, jc)

    d0, d1, d2 = delta[p0], delta[p1], delta[p2]
    b0, b1, b2 = beta[p0], beta[p1], beta[p2]

    m00 = d0 - d2
    m01 = d1 - d2
    m10 = b0 - b2
    m11 = b1 - b2
    det = m00 * m11 - m01 * m10
    if abs(det) < 1e-14:
        return

    rhs0 = -d2
    rhs1 = -b2
    l0 = (m11 * rhs0 - m01 * rhs1) / det
    l1 = (-m10 * rhs0 + m00 * rhs1) / det
    l2 = 1.0 - l0 - l1

    tol = -1e-9
    if l0 < tol or l1 < tol or l2 < tol:
        return
    if l0 > 1 + 1e-9 or l1 > 1 + 1e-9 or l2 > 1 + 1e-9:
        return

    theta0 = np.arctan2(b0, d0)
    theta1 = np.arctan2(b1, d1)
    theta2 = np.arctan2(b2, d2)

    e01 = wrap(theta1 - theta0)
    e12 = wrap(theta2 - theta1)
    e20 = wrap(theta0 - theta2)

    total = e01 + e12 + e20
    n = round(total / (2 * np.pi))

    if n == 1:
        t = 1  # wedge
    elif n == -1:
        t = 0  # trisector
    else:
        return  # ambiguous / higher order, skip

    px = xs[ia] * l0 + xs[ib] * l1 + xs[ic] * l2
    py = ys[ja] * l0 + ys[jb] * l1 + ys[jc] * l2

    points_out.append((px, py, 0.0))
    types_out.append(t)


for j in range(ny - 1):
    for i in range(nx - 1):
        p00 = idx(i, j)
        p10 = idx(i + 1, j)
        p01 = idx(i, j + 1)
        p11 = idx(i + 1, j + 1)
        # skip cells with any masked/invalid data if needed (none here)
        # triangle 1: (i,j),(i+1,j),(i+1,j+1)
        process_triangle(i, j, i + 1, j, i + 1, j + 1)
        # triangle 2: (i,j),(i+1,j+1),(i,j+1)
        process_triangle(i, j, i + 1, j + 1, i, j + 1)

print("Trisectors:", types_out.count(0))
print("Wedges:", types_out.count(1))
print("Total:", len(points_out))

# Build output vtkPolyData
vpoints = vtk.vtkPoints()
for p in points_out:
    vpoints.InsertNextPoint(p)

vtype = vtk.vtkIntArray()
vtype.SetName("Type")
for t in types_out:
    vtype.InsertNextValue(t)

vtypename = vtk.vtkStringArray()
vtypename.SetName("TypeName")
for t in types_out:
    vtypename.InsertNextValue("Wedge" if t == 1 else "Trisector")

polydata = vtk.vtkPolyData()
polydata.SetPoints(vpoints)
polydata.GetPointData().AddArray(vtype)
polydata.GetPointData().AddArray(vtypename)
polydata.GetPointData().SetActiveScalars("Type")

verts = vtk.vtkCellArray()
for k in range(len(points_out)):
    verts.InsertNextCell(1, [k])
polydata.SetVerts(verts)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("degenerate_points.vtp")
writer.SetInputData(polydata)
writer.Write()

print("Wrote degenerate_points.vtp")
