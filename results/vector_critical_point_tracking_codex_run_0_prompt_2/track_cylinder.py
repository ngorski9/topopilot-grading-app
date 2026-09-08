#!/usr/bin/env pvpython
"""Critical-point tracking of the cylinder data using partial optimal transport."""
from pathlib import Path
import csv
import numpy as np
import ot
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

ROOT = Path('/workspace')
FILES = [ROOT/'cylinder'/f'cylinder{i}.vti' for i in (1, 2, 3)]
OUT = ROOT/'cylinder_tracking'
OUT.mkdir(exist_ok=True)

def read_image(fn):
    r = vtk.vtkXMLImageDataReader(); r.SetFileName(str(fn)); r.Update()
    return r.GetOutput()

def critical_points(img):
    nx, ny, _ = img.GetDimensions()
    u = vtk_to_numpy(img.GetPointData().GetArray('u')).reshape(ny, nx)
    v = vtk_to_numpy(img.GetPointData().GetArray('v')).reshape(ny, nx)
    out = []
    # Each vector component is bilinear per image cell. Newton refinement finds
    # its common zero in the unit cell (the PL critical point).
    for j in range(ny-1):
        for i in range(nx-1):
            U = np.array([[u[j,i], u[j,i+1]], [u[j+1,i], u[j+1,i+1]]])
            V = np.array([[v[j,i], v[j,i+1]], [v[j+1,i], v[j+1,i+1]]])
            if not (U.min() <= 0 <= U.max() and V.min() <= 0 <= V.max()):
                continue
            x = y = .5
            for _ in range(12):
                def f(A): return (1-y)*((1-x)*A[0,0]+x*A[0,1])+y*((1-x)*A[1,0]+x*A[1,1])
                fu, fv = f(U), f(V)
                J = np.array([[(1-y)*(U[0,1]-U[0,0])+y*(U[1,1]-U[1,0]),
                               (1-x)*(U[1,0]-U[0,0])+x*(U[1,1]-U[0,1])],
                              [(1-y)*(V[0,1]-V[0,0])+y*(V[1,1]-V[1,0]),
                               (1-x)*(V[1,0]-V[0,0])+x*(V[1,1]-V[0,1])]])
                try: d = np.linalg.solve(J, [-fu, -fv])
                except np.linalg.LinAlgError: break
                x += d[0]; y += d[1]
                if max(abs(d)) < 1e-8: break
            if -.00001 <= x <= 1.00001 and -.00001 <= y <= 1.00001 and abs(f(U))+abs(f(V)) < 1e-6:
                out.append((i+x, j+y, 0.0))
    # Adjacent cells can report the same boundary zero: retain one representative.
    unique=[]
    for p in out:
        if not any(np.linalg.norm(np.subtract(p,q)) < 1e-3 for q in unique): unique.append(p)
    return np.asarray(unique, float).reshape((-1,3))

def partial_matches(a, b):
    if not len(a) or not len(b): return []
    wa, wb = np.ones(len(a))/len(a), np.ones(len(b))/len(b)
    cost = ot.dist(a[:,:2], b[:,:2], metric='sqeuclidean')
    # transport 85% of the maximum mutually available probability mass;
    # remaining points are explicitly allowed to be born/die.
    mass = .85 * min(wa.sum(), wb.sum())
    plan = ot.partial.partial_wasserstein(wa, wb, cost, m=mass)
    pairs=[]; used_a=set(); used_b=set()
    for k in np.argsort(plan.ravel())[::-1]:
        i,j=np.unravel_index(k,plan.shape)
        if plan[i,j] <= 1e-12: break
        if i not in used_a and j not in used_b:
            pairs.append((i,j,float(plan[i,j]))); used_a.add(i); used_b.add(j)
    return pairs

images=[read_image(f) for f in FILES]
cps=[critical_points(im) for im in images]
matches=[partial_matches(cps[0],cps[1]), partial_matches(cps[1],cps[2])]
print('critical point counts:', [len(x) for x in cps])
print('partial-OT links:', [len(x) for x in matches])

# Link only correspondences that survive both transitions; these are the 3-step tracks.
second={i:j for i,j,_ in matches[1]}
tracks=[]
for i,j,_ in matches[0]:
    if j in second: tracks.append((i,j,second[j]))
print('three-step tracks:', len(tracks))

with open(OUT/'critical_points_and_tracks.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['track_id','time_step','point_id','x','y','z'])
    for tid, tr in enumerate(tracks):
        for t,pid in enumerate(tr): w.writerow([tid,t+1,pid,*cps[t][pid]])

# Original vector field, shown as a magnitude-coloured raster plus a sparse arrow glyph field.
base=images[0]
u=vtk_to_numpy(base.GetPointData().GetArray('u')); v=vtk_to_numpy(base.GetPointData().GetArray('v'))
mag=np.sqrt(u*u+v*v); ma=numpy_to_vtk(mag,deep=True); ma.SetName('speed')
base.GetPointData().AddArray(ma); base.GetPointData().SetActiveScalars('speed')
vec=numpy_to_vtk(np.c_[u,v,np.zeros_like(u)],deep=True); vec.SetName('velocity')
base.GetPointData().AddArray(vec); base.GetPointData().SetActiveVectors('velocity')

mask=vtk.vtkMaskPoints(); mask.SetInputData(base); mask.SetOnRatio(28); mask.RandomModeOff(); mask.Update()
arrow=vtk.vtkArrowSource(); arrow.SetTipResolution(8); arrow.SetShaftResolution(8)
glyph=vtk.vtkGlyph3D(); glyph.SetInputConnection(mask.GetOutputPort()); glyph.SetSourceConnection(arrow.GetOutputPort()); glyph.SetVectorModeToUseVector(); glyph.SetScaleModeToScaleByVector(); glyph.SetScaleFactor(9); glyph.OrientOn(); glyph.Update()

def point_poly(points, colors=None):
    p=vtk.vtkPoints(); [p.InsertNextPoint(x) for x in points]
    d=vtk.vtkPolyData(); d.SetPoints(p); verts=vtk.vtkCellArray()
    for i in range(len(points)): verts.InsertNextCell(1); verts.InsertCellPoint(i)
    d.SetVerts(verts)
    if colors is not None:
        a=numpy_to_vtk(np.asarray(colors,np.uint8),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR); a.SetName('RGB'); a.SetNumberOfComponents(3); d.GetPointData().SetScalars(a)
    return d

allpts=np.vstack(cps); cols=np.vstack([np.tile(c,(len(p),1)) for p,c in zip(cps,[(230,60,50),(60,180,75),(65,100,225)])])
cp_poly=point_poly(allpts,cols)
writer=vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(OUT/'tracked_critical_points.vtp')); writer.SetInputData(cp_poly); writer.Write()

pts=vtk.vtkPoints(); lines=vtk.vtkCellArray()
for tr in tracks:
    line=vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(3)
    for k,(t,pid) in enumerate(enumerate(tr)):
        line.GetPointIds().SetId(k,pts.InsertNextPoint(cps[t][pid]))
    lines.InsertNextCell(line)
track_poly=vtk.vtkPolyData(); track_poly.SetPoints(pts); track_poly.SetLines(lines)
writer=vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(OUT/'partial_ot_tracks.vtp')); writer.SetInputData(track_poly); writer.Write()

lut=vtk.vtkLookupTable(); lut.SetHueRange(.667, 0.0); lut.Build()
bm=vtk.vtkDataSetMapper(); bm.SetInputData(base); bm.SetScalarRange(float(mag.min()),float(mag.max())); bm.SetLookupTable(lut)
ba=vtk.vtkActor(); ba.SetMapper(bm)
gm=vtk.vtkPolyDataMapper(); gm.SetInputConnection(glyph.GetOutputPort()); gm.ScalarVisibilityOff()
ga=vtk.vtkActor(); ga.SetMapper(gm); ga.GetProperty().SetColor(.08,.08,.08); ga.GetProperty().SetOpacity(.7)
cm=vtk.vtkPolyDataMapper(); cm.SetInputData(cp_poly); cm.SetScalarModeToUsePointData(); cm.SetColorModeToDirectScalars(); cm.ScalarVisibilityOn()
ca=vtk.vtkActor(); ca.SetMapper(cm); ca.GetProperty().SetPointSize(12)
tm=vtk.vtkPolyDataMapper(); tm.SetInputData(track_poly); tm.ScalarVisibilityOff()
ta=vtk.vtkActor(); ta.SetMapper(tm); ta.GetProperty().SetColor(0,0,0); ta.GetProperty().SetLineWidth(3)
ren=vtk.vtkRenderer(); ren.SetBackground(1,1,1); [ren.AddActor(a) for a in (ba,ga,ta,ca)]
win=vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(1400,850)
ren.ResetCamera(); ren.GetActiveCamera().SetParallelProjection(True); ren.ResetCamera()
win.Render()
png=vtk.vtkWindowToImageFilter(); png.SetInput(win); png.Update(); pw=vtk.vtkPNGWriter(); pw.SetFileName(str(OUT/'cylinder_partial_ot_tracking.png')); pw.SetInputConnection(png.GetOutputPort()); pw.Write()
print('Wrote', OUT)
