import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

r = vtk.vtkXMLImageDataReader()
r.SetFileName('/workspace/brain.vti')
r.Update()
img = r.GetOutput()
nx, ny, nz = img.GetDimensions()
ox, oy, oz = img.GetOrigin()
sx, sy, sz = img.GetSpacing()
pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
D = vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

E1 = A - D
E2 = 2.0 * B

def coord(i, j):
    return ox + i * sx, oy + j * sy

def solve_tri(v0, v1, v2, e1, e2):
    # v* = (x,y); e1,e2 = tuples of scalar values at v0,v1,v2
    (x0, y0), (x1, y1), (x2, y2) = v0, v1, v2
    M = np.array([[1, x0, y0], [1, x1, y1], [1, x2, y2]], dtype=float)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    c1 = Minv @ np.array(e1, dtype=float)  # c0,c1,c2 for E1
    c2 = Minv @ np.array(e2, dtype=float)  # c0,c1,c2 for E2
    # solve c1[1]*x + c1[2]*y = -c1[0]
    #       c2[1]*x + c2[2]*y = -c2[0]
    Jm = np.array([[c1[1], c1[2]], [c2[1], c2[2]]])
    detJ = np.linalg.det(Jm)
    if abs(detJ) < 1e-30:
        return None
    rhs = np.array([-c1[0], -c2[0]])
    xy = np.linalg.solve(Jm, rhs)
    x, y = xy
    # barycentric check
    denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(denom) < 1e-30:
        return None
    l1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / denom
    l2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / denom
    l0 = 1 - l1 - l2
    eps = 1e-9
    if l0 < -eps or l1 < -eps or l2 < -eps:
        return None
    return x, y, detJ

points = []  # (x,y,typ)  typ: 'trisector' or 'wedge'

for j in range(ny - 1):
    for i in range(nx - 1):
        p00 = coord(i, j)
        p10 = coord(i + 1, j)
        p11 = coord(i + 1, j + 1)
        p01 = coord(i, j + 1)

        e1_00, e2_00 = E1[j, i], E2[j, i]
        e1_10, e2_10 = E1[j, i + 1], E2[j, i + 1]
        e1_11, e2_11 = E1[j + 1, i + 1], E2[j + 1, i + 1]
        e1_01, e2_01 = E1[j + 1, i], E2[j + 1, i]

        tris = [
            (p00, p10, p11, (e1_00, e1_10, e1_11), (e2_00, e2_10, e2_11)),
            (p00, p11, p01, (e1_00, e1_11, e1_01), (e2_00, e2_11, e2_01)),
        ]
        for v0, v1, v2, e1s, e2s in tris:
            res = solve_tri(v0, v1, v2, e1s, e2s)
            if res is None:
                continue
            x, y, detJ = res
            typ = 'wedge' if detJ > 0 else 'trisector'
            points.append((x, y, typ))

# dedupe points that are very close (e.g. shared vertices exactly zero, or triangulation seams)
pts_arr = np.array([[p[0], p[1]] for p in points]) if points else np.zeros((0, 2))
keep = []
used = np.zeros(len(points), dtype=bool)
for idx in range(len(points)):
    if used[idx]:
        continue
    x, y, typ = points[idx]
    group = [idx]
    for jdx in range(idx + 1, len(points)):
        if used[jdx]:
            continue
        x2, y2, typ2 = points[jdx]
        if abs(x2 - x) < 1e-6 and abs(y2 - y) < 1e-6:
            group.append(jdx)
            used[jdx] = True
    used[idx] = True
    keep.append(points[idx])

print(f"Total degenerate points found: {len(keep)}")
ntri = sum(1 for p in keep if p[2] == 'trisector')
nwed = sum(1 for p in keep if p[2] == 'wedge')
print(f"trisectors: {ntri}, wedges: {nwed}")

vtk_pts = vtk.vtkPoints()
type_arr = vtk.vtkIntArray()
type_arr.SetName('type')  # 0 = trisector, 1 = wedge
name_arr = vtk.vtkStringArray()
name_arr.SetName('typeName')

for x, y, typ in keep:
    vtk_pts.InsertNextPoint(x, y, 0.0)
    type_arr.InsertNextValue(0 if typ == 'trisector' else 1)
    name_arr.InsertNextValue(typ)

poly = vtk.vtkPolyData()
poly.SetPoints(vtk_pts)
poly.GetPointData().AddArray(type_arr)
poly.GetPointData().AddArray(name_arr)
poly.GetPointData().SetActiveScalars('type')

verts = vtk.vtkVertexGlyphFilter()
verts.SetInputData(poly)
verts.Update()

w = vtk.vtkXMLPolyDataWriter()
w.SetFileName('/workspace/degenerate_points.vtp')
w.SetInputData(verts.GetOutput())
w.Write()
print("Wrote /workspace/degenerate_points.vtp")
