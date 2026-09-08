#!/opt/conda/bin/pvpython
"""Track PL critical points in the cylinder time series and build a ParaView scene.

The image grid is triangulated along a fixed diagonal.  On every triangle u and v
are linear, so their simultaneous zero is found exactly from barycentric weights.
"""
from pathlib import Path
import csv
import numpy as np
import ot
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from paraview.simple import *

ROOT = Path('/workspace')
DATA = ROOT / 'cylinder'
OUT = ROOT / 'cylinder_tracking'
OUT.mkdir(exist_ok=True)
STEPS = [1, 2, 3]

def read_field(step):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(DATA / f'cylinder{step}.vti'))
    reader.Update()
    img = reader.GetOutput()
    nx, ny, _ = img.GetDimensions()
    org, sp = img.GetOrigin(), img.GetSpacing()
    u = vtk_to_numpy(img.GetPointData().GetArray('u')).reshape(ny, nx)
    v = vtk_to_numpy(img.GetPointData().GetArray('v')).reshape(ny, nx)
    return img, nx, ny, org, sp, u, v

def critical_points(nx, ny, org, sp, u, v):
    # Test the two triangles of each pixel: (00,10,11) and (00,11,01).
    pts, kinds = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = [(i,j), (i+1,j), (i+1,j+1), (i,j+1)]
            for ids in ((0,1,2), (0,2,3)):
                ij = [corners[k] for k in ids]
                xy = np.asarray(ij, float)
                uv = np.asarray([[u[y,x], v[y,x]] for x,y in ij], float)
                # uv.T @ lambda = 0 and sum(lambda)=1
                a = np.vstack((uv.T, np.ones(3)))
                try: lam = np.linalg.solve(a, [0., 0., 1.])
                except np.linalg.LinAlgError: continue
                if np.min(lam) < -1.e-8 or np.max(lam) > 1.+1.e-8: continue
                q = lam @ xy
                # J is constant on this PL triangle; determinant distinguishes
                # saddle (negative) from source/sink/center-like (positive).
                mat = np.column_stack((xy[1]-xy[0], xy[2]-xy[0]))
                jac = (uv[1:] - uv[0]).T @ np.linalg.inv(mat)
                kind = 'saddle' if np.linalg.det(jac) < 0 else 'extremum'
                p = np.array([org[0] + sp[0]*q[0], org[1] + sp[1]*q[1], 0.])
                # A zero lying precisely on the fixed diagonal can be found twice.
                if not any(np.linalg.norm(p[:2]-old[:2]) < 1.e-6 for old in pts):
                    pts.append(p); kinds.append(kind)
    return np.asarray(pts), kinds

def write_points(path, points, steps, track_ids, kinds):
    poly = vtk.vtkPolyData(); vp = vtk.vtkPoints()
    for p in points: vp.InsertNextPoint(*p)
    poly.SetPoints(vp)
    verts = vtk.vtkCellArray()
    for i in range(len(points)):
        verts.InsertNextCell(1); verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    for name, vals in [('time_step', steps), ('track_id', track_ids)]:
        arr = vtk.vtkIntArray(); arr.SetName(name)
        for x in vals: arr.InsertNextValue(int(x))
        poly.GetPointData().AddArray(arr)
    arr = vtk.vtkStringArray(); arr.SetName('critical_type')
    for x in kinds: arr.InsertNextValue(x)
    poly.GetPointData().AddArray(arr)
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(poly); w.Write()

def write_tracks(path, trajectories):
    poly = vtk.vtkPolyData(); vp = vtk.vtkPoints(); lines = vtk.vtkCellArray()
    tidarr = vtk.vtkIntArray(); tidarr.SetName('track_id')
    for tid, seq in trajectories.items():
        if len(seq) < 2: continue
        line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(len(seq))
        for k, (_, p) in enumerate(seq):
            idx = vp.InsertNextPoint(*p); line.GetPointIds().SetId(k, idx)
        lines.InsertNextCell(line); tidarr.InsertNextValue(tid)
    poly.SetPoints(vp); poly.SetLines(lines); poly.GetCellData().AddArray(tidarr)
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(poly); w.Write()

fields, cps, types = [], [], []
for step in STEPS:
    fields.append(read_field(step))
    _, nx, ny, org, sp, u, v = fields[-1]
    p, k = critical_points(nx, ny, org, sp, u, v)
    cps.append(p); types.append(k)
    print(f'time step {step}: {len(p)} critical points')

# Partial OT supplies the correspondences.  Transport only 90% of uniform mass,
# leaving unmatched births/deaths available rather than forcing bad long matches.
ids = [list(range(len(cps[0])))]
next_id = len(cps[0]); links = []
for t in range(2):
    a = np.ones(len(cps[t])) / max(len(cps[t]), 1)
    b = np.ones(len(cps[t+1])) / max(len(cps[t+1]), 1)
    cost = ot.dist(cps[t][:,:2], cps[t+1][:,:2], metric='sqeuclidean')
    gamma = ot.partial.partial_wasserstein(a, b, cost, m=0.90)
    # Convert possibly fractional OT plan to one-to-one links in descending flow.
    chosen_a, chosen_b = set(), set(); newids = [-1] * len(cps[t+1])
    for i,j in sorted(zip(*np.nonzero(gamma > 1.e-12)), key=lambda ij: gamma[ij], reverse=True):
        if i in chosen_a or j in chosen_b: continue
        chosen_a.add(i); chosen_b.add(j); newids[j] = ids[t][i]
        links.append((STEPS[t], i, STEPS[t+1], j, float(gamma[i,j]), float(np.sqrt(cost[i,j]))))
    for j in range(len(newids)):
        if newids[j] < 0: newids[j] = next_id; next_id += 1
    ids.append(newids)

all_points = np.vstack(cps); all_steps = np.concatenate([[s]*len(p) for s,p in zip(STEPS,cps)])
all_ids = np.concatenate([np.asarray(x) for x in ids]); all_types = sum(types, [])
write_points(OUT/'critical_points.vtp', all_points, all_steps, all_ids, all_types)
trajectories = {}
for t in range(3):
    for tid,p in zip(ids[t], cps[t]): trajectories.setdefault(tid, []).append((STEPS[t],p))
write_tracks(OUT/'trajectories.vtp', trajectories)
with open(OUT/'transport_links.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['from_step','from_point','to_step','to_point','transport_mass','distance'])
    wr.writerows(links)

# ParaView visual output: field at step 3, stream glyphs, tracked critical points,
# and partial-OT trajectories.  The state file keeps the complete reproducible scene.
reader = XMLImageDataReader(FileName=[str(DATA/'cylinder3.vti')])
reader.PointArrayStatus = ['u', 'v']
reader.UpdatePipeline()
calc = Calculator(Input=reader); calc.ResultArrayName='velocity'; calc.Function='u*iHat + v*jHat'
calc.UpdatePipeline()
glyph = Glyph(Input=calc, GlyphType='Arrow'); glyph.OrientationArray=['POINTS','velocity']; glyph.ScaleArray=['POINTS','velocity']; glyph.ScaleFactor=7.0; glyph.GlyphMode='Every Nth Point'; glyph.Stride=100
points = XMLPolyDataReader(FileName=[str(OUT/'critical_points.vtp')])
tracks = XMLPolyDataReader(FileName=[str(OUT/'trajectories.vtp')])
view = GetActiveViewOrCreate('RenderView'); view.ViewSize=[1400,900]; view.InteractionMode='2D'; view.Background=[1,1,1]
Show(calc,view).Visibility=0
gdisp=Show(glyph,view); gdisp.SetScalarColoring(None, 0); gdisp.DiffuseColor=[0.18,0.35,0.7]; gdisp.Opacity=0.7
pdisp=Show(points,view); pdisp.SetScalarColoring(None, 0); pdisp.Representation='Points'; pdisp.PointSize=9; pdisp.DiffuseColor=[0.95,0.05,0.05]
tdisp=Show(tracks,view); tdisp.SetScalarColoring(None, 0); tdisp.LineWidth=5; tdisp.DiffuseColor=[0.05,0.8,0.12]
Render(); ResetCamera(view); Render()
SaveScreenshot(str(OUT/'cylinder_critical_tracking.png'), view)
SaveState(str(OUT/'cylinder_critical_tracking.pvsm'))
print(f'Wrote {OUT}')
