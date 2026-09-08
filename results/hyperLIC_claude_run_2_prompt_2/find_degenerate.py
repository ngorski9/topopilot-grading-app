import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('/workspace/brain.vti')
reader.Update()
img = reader.GetOutput()

ext = img.GetExtent()
nx = ext[1] - ext[0] + 1
ny = ext[3] - ext[2] + 1
origin = img.GetOrigin()
spacing = img.GetSpacing()

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
D = vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

# double-angle vector field
d1 = A - D
d2 = 2.0 * B

def coord(i, j):
    x = origin[0] + i * spacing[0]
    y = origin[1] + j * spacing[1]
    return x, y

def vec(i, j):
    return d1[j, i], d2[j, i]

def classify_triangle(v0, v1, v2):
    # solve w0*v0+w1*v1+w2*v2=0, w0+w1+w2=1
    M = np.array([
        [v0[0], v1[0], v2[0]],
        [v0[1], v1[1], v2[1]],
        [1.0, 1.0, 1.0]
    ])
    rhs = np.array([0.0, 0.0, 1.0])
    try:
        w = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    tol = 1e-9
    if np.all(w >= -tol) and np.all(w <= 1 + tol):
        # winding number of direction angle around triangle
        thetas = [np.arctan2(v[1], v[0]) for v in (v0, v1, v2)]
        total = 0.0
        for k in range(3):
            dth = thetas[(k + 1) % 3] - thetas[k]
            # wrap to (-pi, pi]
            while dth > np.pi:
                dth -= 2 * np.pi
            while dth <= -np.pi:
                dth += 2 * np.pi
            total += dth
        index_raw = round(total / (2 * np.pi))
        if index_raw == 0:
            return None
        return w, index_raw
    return None

points = []  # (x,y,z,type)  type: 'wedge' or 'trisector'

for j in range(ny - 1):
    for i in range(nx - 1):
        # quad corners: (i,j) (i+1,j) (i+1,j+1) (i,j+1)
        p00 = coord(i, j)
        p10 = coord(i + 1, j)
        p11 = coord(i + 1, j + 1)
        p01 = coord(i, j + 1)
        v00 = vec(i, j)
        v10 = vec(i + 1, j)
        v11 = vec(i + 1, j + 1)
        v01 = vec(i, j + 1)

        tris = [
            ((p00, p10, p11), (v00, v10, v11)),
            ((p00, p11, p01), (v00, v11, v01)),
        ]
        for (pa, pb, pc), (va, vb, vc) in tris:
            res = classify_triangle(va, vb, vc)
            if res is None:
                continue
            w, index_raw = res
            x = w[0] * pa[0] + w[1] * pb[0] + w[2] * pc[0]
            y = w[0] * pa[1] + w[1] * pb[1] + w[2] * pc[1]
            ptype = 'wedge' if index_raw == 1 else ('trisector' if index_raw == -1 else 'other')
            points.append((x, y, 0.0, ptype))

print('found', len(points), 'degenerate points')
n_wedge = sum(1 for p in points if p[3] == 'wedge')
n_tri = sum(1 for p in points if p[3] == 'trisector')
n_other = sum(1 for p in points if p[3] == 'other')
print('wedges:', n_wedge, 'trisectors:', n_tri, 'other:', n_other)

with open('/workspace/degenerate_points.csv', 'w') as f:
    f.write('x,y,z,type\n')
    for x, y, z, t in points:
        f.write(f'{x},{y},{z},{t}\n')

with open('/workspace/wedge_points.csv', 'w') as f:
    f.write('x,y,z\n')
    for x, y, z, t in points:
        if t == 'wedge':
            f.write(f'{x},{y},{z}\n')

with open('/workspace/trisector_points.csv', 'w') as f:
    f.write('x,y,z\n')
    for x, y, z, t in points:
        if t == 'trisector':
            f.write(f'{x},{y},{z}\n')
