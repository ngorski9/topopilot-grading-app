from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy

root = Path(__file__).resolve().parent
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(str(root / "brain.vti"))
reader.Update()
image = reader.GetOutput()
pd = image.GetPointData()
a = vtk_to_numpy(pd.GetArray("A"))
b = vtk_to_numpy(pd.GetArray("B"))
d = vtk_to_numpy(pd.GetArray("D"))
dims = image.GetDimensions()
nx, ny = dims[0], dims[1]

# For a symmetric 2-D tensor [[A,B],[B,D]], degeneracy means A-D=0 and B=0.
# On each PL triangle these are two affine functions, so their common zero is exact.
field0 = a - d
field1 = b
classes = {"Wedges": [], "Trisectors": []}
seen = set()

def add_triangle(ids):
    p = [image.GetPoint(i) for i in ids]
    f0 = [field0[i] for i in ids]
    f1 = [field1[i] for i in ids]
    # f(x,y) = f0 + J * (x-x0,y-y0)
    x0, y0, _ = p[0]
    m00, m01 = p[1][0]-x0, p[2][0]-x0
    m10, m11 = p[1][1]-y0, p[2][1]-y0
    detm = m00*m11 - m01*m10
    if abs(detm) < 1e-14:
        return
    inv00, inv01 = m11/detm, -m01/detm
    inv10, inv11 = -m10/detm, m00/detm
    df00, df01 = f0[1]-f0[0], f0[2]-f0[0]
    df10, df11 = f1[1]-f1[0], f1[2]-f1[0]
    j00, j01 = df00*inv00 + df01*inv10, df00*inv01 + df01*inv11
    j10, j11 = df10*inv00 + df11*inv10, df10*inv01 + df11*inv11
    detj = j00*j11 - j01*j10
    if abs(detj) < 1e-14:
        return
    dx = (-f0[0]*j11 + f1[0]*j01) / detj
    dy = (-j00*f1[0] + j10*f0[0]) / detj
    # Barycentric containment, permitting only numerical tolerance at cell edges.
    lam1 = (dx*m11 - dy*m01) / detm
    lam2 = (m00*dy - m10*dx) / detm
    lam0 = 1.0-lam1-lam2
    if min(lam0, lam1, lam2) < -1e-9:
        return
    x, y = x0+dx, y0+dy
    key = (round(x, 9), round(y, 9))
    if key in seen:
        return
    seen.add(key)
    # Positive/negative index respectively identify wedge/trisector singularities.
    classes["Wedges" if detj > 0 else "Trisectors"].append((x, y, p[0][2]))

for y in range(ny-1):
    for x in range(nx-1):
        q0 = y*nx+x
        q1, q2, q3 = q0+1, q0+nx, q0+nx+1
        add_triangle((q0, q1, q3))
        add_triangle((q0, q3, q2))

for name, points in classes.items():
    poly = vtk.vtkPolyData()
    vtkpts = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for point in points:
        pid = vtkpts.InsertNextPoint(point)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
    poly.SetPoints(vtkpts)
    poly.SetVerts(verts)
    arr = vtk.vtkStringArray()
    arr.SetName("DegenerateType")
    for _ in points:
        arr.InsertNextValue(name[:-1])
    poly.GetPointData().AddArray(arr)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(root / f"brain_{name.lower()}.vtp"))
    writer.SetInputData(poly)
    writer.Write()
    print(f"{name}: {len(points)}")
