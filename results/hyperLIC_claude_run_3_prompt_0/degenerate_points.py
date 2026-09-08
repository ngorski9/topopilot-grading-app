"""
Compute degenerate points (wedges and trisectors) of a 2D symmetric
piecewise-linear tensor field T = [[A, B], [B, D]] stored in brain.vti.

Algorithm (Delmarcelle & Hesselink, 1994): degenerate points are the zeros
of the tensor deviator (E, F) = ((A - D) / 2, B). Each pixel quad is split
into two triangles; inside every triangle the field is linear, so any zero
of (E, F) is found by solving a barycentric linear system. The zero is
classified as a wedge (index +1/2) if the local Jacobian determinant of
(E, F) with respect to (x, y) is positive, and a trisector (index -1/2) if
it is negative.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("brain.vti")
reader.Update()
img = reader.GetOutput()

dims = img.GetDimensions()
nx, ny, nz = dims
origin = img.GetOrigin()
spacing = img.GetSpacing()
print("dims", dims, "origin", origin, "spacing", spacing)

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray("A")).reshape((nz, ny, nx))[0]
B = vtk_to_numpy(pd.GetArray("B")).reshape((nz, ny, nx))[0]
D = vtk_to_numpy(pd.GetArray("D")).reshape((nz, ny, nx))[0]

E = 0.5 * (A - D)
F = B.copy()

wedges = []
trisectors = []

def solve_triangle(p0, p1, p2, e0, e1, e2, f0, f1, f2):
    # barycentric system for l0, l1 (l2 = 1 - l0 - l1)
    m = np.array([[e0 - e2, e1 - e2],
                  [f0 - f2, f1 - f2]])
    rhs = np.array([-e2, -f2])
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if abs(det) < 1e-14:
        return None
    l0 = (rhs[0] * m[1, 1] - m[0, 1] * rhs[1]) / det
    l1 = (m[0, 0] * rhs[1] - rhs[0] * m[1, 0]) / det
    l2 = 1.0 - l0 - l1
    if l0 < -1e-9 or l1 < -1e-9 or l2 < -1e-9:
        return None
    pos = l0 * np.array(p0) + l1 * np.array(p1) + l2 * np.array(p2)

    M = np.array([[p1[0] - p0[0], p1[1] - p0[1]],
                  [p2[0] - p0[0], p2[1] - p0[1]]])
    dE = np.array([e1 - e0, e2 - e0])
    dF = np.array([f1 - f0, f2 - f0])
    try:
        gradE = np.linalg.solve(M, dE)
        gradF = np.linalg.solve(M, dF)
    except np.linalg.LinAlgError:
        return None
    delta = gradE[0] * gradF[1] - gradE[1] * gradF[0]
    return pos, delta


for j in range(ny - 1):
    for i in range(nx - 1):
        p00 = (i, j)
        p10 = (i + 1, j)
        p11 = (i + 1, j + 1)
        p01 = (i, j + 1)

        e00, e10, e11, e01 = E[j, i], E[j, i + 1], E[j + 1, i + 1], E[j + 1, i]
        f00, f10, f11, f01 = F[j, i], F[j, i + 1], F[j + 1, i + 1], F[j + 1, i]

        for (pa, pb, pc, ea, eb, ec, fa, fb, fc) in (
            (p00, p10, p11, e00, e10, e11, f00, f10, f11),
            (p00, p11, p01, e00, e11, e01, f00, f11, f01),
        ):
            res = solve_triangle(pa, pb, pc, ea, eb, ec, fa, fb, fc)
            if res is None:
                continue
            pos, delta = res
            pt = (pos[0] + origin[0], pos[1] + origin[1], 0.0)
            if delta > 0:
                wedges.append(pt)
            else:
                trisectors.append(pt)

print("wedges:", len(wedges), "trisectors:", len(trisectors))


def write_points(points, filename):
    vpoints = vtk.vtkPoints()
    for p in points:
        vpoints.InsertNextPoint(p)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vpoints)
    verts = vtk.vtkCellArray()
    for idx in range(len(points)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(idx)
    poly.SetVerts(verts)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly)
    writer.Write()


write_points(wedges, "wedges.vtp")
write_points(trisectors, "trisectors.vtp")

# Also write an augmented copy of the field with a full 3x3 tensor array
# (zero-padded in z) so ParaView can glyph / color it as a tensor field.
n = nx * ny
tensor = np.zeros((n, 9), dtype=np.float64)
Af = A.ravel(order="C")
Bf = B.ravel(order="C")
Df = D.ravel(order="C")
tensor[:, 0] = Af  # xx
tensor[:, 1] = Bf  # xy
tensor[:, 3] = Bf  # yx
tensor[:, 4] = Df  # yy
vtk_tensor = numpy_to_vtk(tensor, deep=True)
vtk_tensor.SetName("Tensor")
img.GetPointData().AddArray(vtk_tensor)

writer2 = vtk.vtkXMLImageDataWriter()
writer2.SetFileName("brain_with_tensor.vti")
writer2.SetInputData(img)
writer2.Write()

print("done")
