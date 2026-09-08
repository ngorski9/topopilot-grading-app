#!/usr/bin/env /opt/conda/bin/pvpython
"""Extract 2-D PL vector-field critical points and link them with partial OT."""
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot

ROOT = Path('/workspace')
DATA = ROOT / 'cylinder'
STEPS = [1, 11, 21]

def read(step):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(DATA / f'cylinder{step}.vti'))
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()
    u = vtk_to_numpy(image.GetPointData().GetArray('u')).reshape((dims[1], dims[0]))
    v = vtk_to_numpy(image.GetPointData().GetArray('v')).reshape((dims[1], dims[0]))
    return u, v

def critical_points(u, v):
    """Roots of the linearly interpolated (u,v) field in two triangles / pixel."""
    ny, nx = u.shape
    pts = []
    kinds = []
    # CCW triangle ordering matches Cartesian x/y coordinates.
    for y in range(ny - 1):
        for x in range(nx - 1):
            corners = [(x,y), (x+1,y), (x+1,y+1), (x,y+1)]
            for tri in ((0,1,2), (0,2,3)):
                xy = np.array([corners[i] for i in tri], dtype=float)
                vec = np.array([[
                    u[y + (i in (2, 3)), x + (i in (1, 2))],
                    v[y + (i in (2, 3)), x + (i in (1, 2))]
                ] for i in tri])
                # v0 + [v1-v0 v2-v0] * barycentric coordinates
                mat = np.column_stack((vec[1] - vec[0], vec[2] - vec[0]))
                if abs(np.linalg.det(mat)) < 1e-12:
                    continue
                ab = np.linalg.solve(mat, -vec[0])
                if ab[0] >= -1e-9 and ab[1] >= -1e-9 and ab.sum() <= 1 + 1e-9:
                    p = xy[0] + ab[0] * (xy[1] - xy[0]) + ab[1] * (xy[2] - xy[0])
                    jac = np.column_stack(((vec[1]-vec[0]), (vec[2]-vec[0]))) @ np.linalg.inv(np.column_stack((xy[1]-xy[0], xy[2]-xy[0])))
                    eig = np.linalg.eigvals(jac)
                    kind = 0 if np.linalg.det(jac) < 0 else (1 if np.iscomplex(eig[0]) else 2)
                    pts.append((p[0], p[1], 0.0)); kinds.append(kind) # saddle, center, node
    return np.asarray(pts), np.asarray(kinds, dtype=np.int32)

def write_points(all_pts, all_kinds):
    poly = vtk.vtkPolyData(); points = vtk.vtkPoints(); verts = vtk.vtkCellArray()
    times = vtk.vtkIntArray(); times.SetName('TimeStep')
    kinds = vtk.vtkIntArray(); kinds.SetName('CriticalType')
    for ti, (arr, typ) in enumerate(zip(all_pts, all_kinds)):
        for p, k in zip(arr, typ):
            pid = points.InsertNextPoint(*p); verts.InsertNextCell(1); verts.InsertCellPoint(pid)
            times.InsertNextValue(STEPS[ti]); kinds.InsertNextValue(int(k))
    poly.SetPoints(points); poly.SetVerts(verts); poly.GetPointData().AddArray(times); poly.GetPointData().AddArray(kinds)
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(ROOT/'tracked_critical_points.vtp')); w.SetInputData(poly); w.Write()

def write_tracks(all_pts, links):
    poly = vtk.vtkPolyData(); points = vtk.vtkPoints(); lines = vtk.vtkCellArray()
    tids = vtk.vtkIntArray(); tids.SetName('TrackId'); tid = 0
    # chain only links that persist in both intervals
    first, second = links
    next_for = {i:j for i,j in first}; final_for = {i:j for i,j in second}
    for i, j in next_for.items():
        if j not in final_for: continue
        k = final_for[j]
        line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(3)
        for q, p in enumerate((all_pts[0][i], all_pts[1][j], all_pts[2][k])):
            line.GetPointIds().SetId(q, points.InsertNextPoint(*p))
        lines.InsertNextCell(line); tids.InsertNextValue(tid); tid += 1
    # retain partial two-step links too, so matched transport is visible
    for interval, pairset in enumerate(links):
        for i,j in pairset:
            if interval == 0 and i in next_for and j in final_for: continue
            line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(2)
            for q, p in enumerate((all_pts[interval][i], all_pts[interval+1][j])):
                line.GetPointIds().SetId(q, points.InsertNextPoint(*p))
            lines.InsertNextCell(line); tids.InsertNextValue(tid); tid += 1
    poly.SetPoints(points); poly.SetLines(lines); poly.GetCellData().AddArray(tids)
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(ROOT/'partial_ot_tracks.vtp')); w.SetInputData(poly); w.Write()

all_pts=[]; all_kinds=[]
for s in STEPS:
    p,k=critical_points(*read(s)); all_pts.append(p); all_kinds.append(k)

links=[]
for a,b in zip(all_pts[:-1], all_pts[1:]):
    if not len(a) or not len(b): links.append([]); continue
    cost = ot.dist(a[:,:2], b[:,:2], metric='sqeuclidean')
    # Transport 90% of the smaller population: remaining mass represents births/deaths.
    gamma = ot.partial.partial_wasserstein(np.ones(len(a))/len(a), np.ones(len(b))/len(b), cost,
                                           m=0.90*min(1.0, len(a)/len(b), len(b)/len(a)))
    # Partial OT can split a source mass. Convert its support into an explicit
    # one-to-one tracking relation, preserving the strongest transport first.
    pairs=[]; used_a=set(); used_b=set()
    candidates = sorted(zip(*np.where(gamma > 1e-10)), key=lambda ij: gamma[ij[0], ij[1]], reverse=True)
    for i,j in candidates:
        if i not in used_a and j not in used_b and cost[i,j] <= 30.0**2:
            pairs.append((int(i),int(j))); used_a.add(i); used_b.add(j)
    links.append(pairs)

write_points(all_pts, all_kinds); write_tracks(all_pts, links)
(ROOT/'cylinder_3steps.pvd').write_text('''<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>\n'''+''.join(f'  <DataSet timestep="{i}" group="" part="0" file="cylinder/cylinder{s}.vti"/>\n' for i,s in enumerate(STEPS))+'''</Collection></VTKFile>\n''')
print('Critical points:', [len(x) for x in all_pts])
print('Partial-OT links:', [len(x) for x in links])
