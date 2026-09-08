"""Critical-point tracking for the cylinder vector-field time series."""
from pathlib import Path
import csv
import numpy as np
import ot
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkPolyLine
from paraview.simple import *

ROOT = Path('/workspace')
DATA = ROOT / 'cylinder'
OUT = ROOT / 'cylinder_tracking_output'
OUT.mkdir(exist_ok=True)

def read_field(step):
    r = vtkXMLImageDataReader(); r.SetFileName(str(DATA / f'cylinder{step}.vti')); r.Update()
    im = r.GetOutput(); dims = im.GetDimensions()
    # VTK image point ordering is x-fastest; reshape as (y, x).
    u = vtk_to_numpy(im.GetPointData().GetArray('u')).reshape(dims[1], dims[0])
    v = vtk_to_numpy(im.GetPointData().GetArray('v')).reshape(dims[1], dims[0])
    return im, u, v

def critical_points(u, v):
    """All roots of the bilinear interpolant in cells that can contain a root."""
    ny, nx = u.shape; roots = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            uu = u[j:j+2, i:i+2]; vv = v[j:j+2, i:i+2]
            if uu.min() > 0 or uu.max() < 0 or vv.min() > 0 or vv.max() < 0: continue
            # Newton solve in local cell coordinates.  Bilinear coefficients.
            au = np.array([uu[0,0], uu[0,1]-uu[0,0], uu[1,0]-uu[0,0], uu[1,1]-uu[0,1]-uu[1,0]+uu[0,0]])
            av = np.array([vv[0,0], vv[0,1]-vv[0,0], vv[1,0]-vv[0,0], vv[1,1]-vv[0,1]-vv[1,0]+vv[0,0]])
            x = np.array([.5, .5])
            for _ in range(12):
                X,Y=x; f=np.array([au[0]+au[1]*X+au[2]*Y+au[3]*X*Y, av[0]+av[1]*X+av[2]*Y+av[3]*X*Y])
                J=np.array([[au[1]+au[3]*Y, au[2]+au[3]*X],[av[1]+av[3]*Y, av[2]+av[3]*X]])
                try: dx=np.linalg.solve(J, f)
                except np.linalg.LinAlgError: break
                x -= dx
                if np.linalg.norm(dx) < 1e-10: break
            X,Y=x
            # Exclude the zero-valued solid-cylinder interior: it is a
            # degenerate continuum, not an isolated vector-field critical point.
            det = np.linalg.det(np.array([[au[1]+au[3]*Y, au[2]+au[3]*X], [av[1]+av[3]*Y, av[2]+av[3]*X]]))
            if abs(det) > 1e-7 and -1e-7 <= X <= 1+1e-7 and -1e-7 <= Y <= 1+1e-7:
                p=np.array([i+X,j+Y])
                if not any(np.linalg.norm(p-q) < 1e-4 for q in roots): roots.append(p)
    return np.array(roots)

images=[]; cps=[]
for t in (1,2,3):
    im,u,v=read_field(t); images.append(im); cps.append(critical_points(u,v))
    print(f'time {t}: {len(cps[-1])} critical points')

# Partial optimal transport: 95% of normalized mass is allowed to move.  The
# resulting transport plan is converted to maximum-flow one-to-one links.
links=[]; plans=[]
for t in range(2):
    a=np.full(len(cps[t]),1/len(cps[t])); b=np.full(len(cps[t+1]),1/len(cps[t+1]))
    M=ot.dist(cps[t],cps[t+1], metric='sqeuclidean'); M/=max(M.max(), 1.0)
    G=ot.partial.partial_wasserstein(a,b,M,m=0.95)
    plans.append(G)
    used_a=set(); used_b=set()
    for ii,jj in np.dstack(np.unravel_index(np.argsort(G.ravel())[::-1],G.shape))[0]:
        if G[ii,jj] <= 1e-12: break
        if ii not in used_a and jj not in used_b:
            links.append((t,int(ii),t+1,int(jj),float(G[ii,jj])))
            used_a.add(int(ii)); used_b.add(int(jj))
    print(f'partial OT {t+1}->{t+2}: {len(used_a)} links, transported mass {G.sum():.3f}')

with open(OUT/'critical_points_and_tracks.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['time_step','point_id','x','y'])
    for t,pts in enumerate(cps,1):
        for k,p in enumerate(pts): w.writerow([t,k,*p])
with open(OUT/'partial_ot_links.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['time_from','id_from','time_to','id_to','transport_mass'])
    for row in links:w.writerow([row[0]+1,row[1],row[2]+1,row[3],row[4]])

# Build point and polyline VTK output (z encodes a tiny time separation).
allpts=vtkPoints(); colors=vtkUnsignedCharArray(); colors.SetName('TimeColor'); colors.SetNumberOfComponents(3)
vtk_ids={}; pal=[(255,70,70),(80,230,100),(70,150,255)]
for t,pts in enumerate(cps):
    for k,p in enumerate(pts):
        vtk_ids[t,k]=allpts.InsertNextPoint(float(p[0]),float(p[1]),0.5+t*0.2); colors.InsertNextTuple3(*pal[t])
cp_poly=vtkPolyData(); cp_poly.SetPoints(allpts); cp_poly.GetPointData().SetScalars(colors)
verts=vtkCellArray()
for n in range(allpts.GetNumberOfPoints()): verts.InsertNextCell(1); verts.InsertCellPoint(n)
cp_poly.SetVerts(verts)
w=vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/'critical_points.vtp')); w.SetInputData(cp_poly); w.Write()

track_poly=vtkPolyData(); track_poly.SetPoints(allpts); lines=vtkCellArray()
# stitch consecutive links into the displayed segments; valid OT links are the tracks.
for t,i,tn,j,m in links:
    line=vtkPolyLine(); line.GetPointIds().SetNumberOfIds(2); line.GetPointIds().SetId(0,vtk_ids[t,i]); line.GetPointIds().SetId(1,vtk_ids[tn,j]); lines.InsertNextCell(line)
track_poly.SetLines(lines); w=vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/'partial_ot_tracks.vtp')); w.SetInputData(track_poly); w.Write()

# Visualization: original t=1 field as arrows, with all tracked points/links overlaid.
src=XMLImageDataReader(FileName=[str(DATA/'cylinder1.vti')]);
calc=Calculator(Input=src); calc.ResultArrayName='velocity'; calc.Function='u*iHat + v*jHat'; calc.AttributeType='Point Data'
norm=Calculator(Input=calc); norm.ResultArrayName='unit_velocity'; norm.Function='velocity / mag(velocity)'; norm.AttributeType='Point Data'
mask=MaskPoints(Input=norm); mask.OnRatio=120
glyph=Glyph(Input=mask, GlyphType='Arrow'); glyph.OrientationArray=['POINTS','unit_velocity']; glyph.ScaleArray=['POINTS','unit_velocity']; glyph.ScaleFactor=7.0; glyph.GlyphMode='All Points'
gdisp=Show(glyph); gdisp.DiffuseColor=[0.08,0.20,0.45]
pd=XMLPolyDataReader(FileName=[str(OUT/'critical_points.vtp')]); pdisp=Show(pd); pdisp.Representation='Points'; pdisp.PointSize=11; pdisp.DiffuseColor=[0.9,0.05,0.08]
td=XMLPolyDataReader(FileName=[str(OUT/'partial_ot_tracks.vtp')]); tdisp=Show(td); tdisp.DiffuseColor=[1.0,0.82,0.05]; tdisp.LineWidth=5
view=GetActiveViewOrCreate('RenderView'); view.ViewSize=[1200,700]; view.UseColorPaletteForBackground=0; view.Background=[1,1,1]; view.InteractionMode='2D'; view.CameraParallelProjection=1
view.CameraPosition=[75,225,700]; view.CameraFocalPoint=[75,225,0]; view.CameraParallelScale=260
Render(); SaveScreenshot(str(OUT/'cylinder_partial_ot_tracking.png'),view)
SaveState(str(OUT/'cylinder_partial_ot_tracking.pvsm'))
print('Wrote',OUT)
