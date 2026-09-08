"""Build a ParaView scene for partial-OT critical-point tracking."""
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot

ROOT = "/workspace"
DATA = os.path.join(ROOT, "cylinder")
OUT = os.path.join(ROOT, "cylinder_tracking")
os.makedirs(OUT, exist_ok=True)
steps = [1, 11, 21]  # three representative temporal samples

def critical_points(path):
    r = vtk.vtkXMLImageDataReader(); r.SetFileName(path); r.Update()
    im = r.GetOutput(); ex = im.GetExtent(); nx = ex[1]-ex[0]+1; ny = ex[3]-ex[2]+1
    u = vtk_to_numpy(im.GetPointData().GetArray("u")).reshape(ny,nx)
    v = vtk_to_numpy(im.GetPointData().GetArray("v")).reshape(ny,nx)
    pts=[]
    # The two fixed triangles per pixel cell give the intended piecewise-linear field.
    for j in range(ny-1):
      for i in range(nx-1):
        for tri in (((0,0),(1,0),(1,1)), ((0,0),(1,1),(0,1))):
          xy=np.asarray(tri,float)
          fv=np.array([[u[j+dy,i+dx],v[j+dy,i+dx]] for dx,dy in tri])
          A=np.column_stack((fv[1]-fv[0],fv[2]-fv[0]))
          try: w=np.linalg.solve(A,-fv[0])
          except np.linalg.LinAlgError: continue
          bary=np.array([1-w.sum(),w[0],w[1]])
          if bary.min() >= -1e-8 and bary.max() <= 1+1e-8:
            p=bary @ xy + [i,j]
            J=np.column_stack((fv[1]-fv[0],fv[2]-fv[0])) @ np.linalg.inv(np.column_stack((xy[1]-xy[0],xy[2]-xy[0])))
            det=np.linalg.det(J); tr=np.trace(J)
            typ=2 if det < 0 else (1 if tr < 0 else 0) # saddle, sink, source
            pts.append((p[0],p[1],typ))
    # eliminate shared-edge duplicates
    unique=[]
    for q in pts:
      if not any(np.hypot(q[0]-z[0],q[1]-z[1]) < 1e-5 for z in unique): unique.append(q)
    return np.asarray(unique,float)

sets=[critical_points(os.path.join(DATA,f"cylinder{s}.vti")) for s in steps]
print("critical points", [len(x) for x in sets])

# Partial OT: retain 85% of the smaller uniform mass, so unmatched births/deaths are allowed.
links=[]
for k,(a,b) in enumerate(zip(sets[:-1],sets[1:])):
  if len(a)==0 or len(b)==0: continue
  M=ot.dist(a[:,:2],b[:,:2],metric='euclidean'); M/=max(float(M.max()),1.0)
  wa=np.ones(len(a))/len(a); wb=np.ones(len(b))/len(b)
  G=ot.partial.partial_wasserstein(wa,wb,M,m=0.85,nb_dummies=max(len(a),len(b)))
  for i in range(len(a)):
    j=int(np.argmax(G[i])); mass=G[i,j]
    # retain meaningful sparse transport assignments, preserving type when possible
    if mass > 1e-9 and a[i,2] == b[j,2]: links.append((k,i,k+1,j,float(mass)))
print("partial-OT links",len(links))

def write_points():
  po=vtk.vtkPoints(); verts=vtk.vtkCellArray(); tarr=vtk.vtkIntArray(); tarr.SetName("TimeStep")
  carr=vtk.vtkIntArray(); carr.SetName("CriticalType")
  for k,S in enumerate(sets):
    for p in S:
      n=po.InsertNextPoint(float(p[0]),float(p[1]),0.0); verts.InsertNextCell(1); verts.InsertCellPoint(n)
      tarr.InsertNextValue(k); carr.InsertNextValue(int(p[2]))
  pd=vtk.vtkPolyData(); pd.SetPoints(po); pd.SetVerts(verts); pd.GetPointData().AddArray(tarr); pd.GetPointData().AddArray(carr)
  w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(os.path.join(OUT,"critical_points.vtp")); w.SetInputData(pd); w.Write()
def write_tracks():
  po=vtk.vtkPoints(); lines=vtk.vtkCellArray(); mass=vtk.vtkDoubleArray(); mass.SetName("OTMass")
  for k,i,l,j,m in links:
    line=vtk.vtkLine(); line.GetPointIds().SetId(0,po.InsertNextPoint(*sets[k][i,:2],0.2)); line.GetPointIds().SetId(1,po.InsertNextPoint(*sets[l][j,:2],0.2)); lines.InsertNextCell(line); mass.InsertNextValue(m)
  pd=vtk.vtkPolyData(); pd.SetPoints(po); pd.SetLines(lines); pd.GetCellData().AddArray(mass)
  w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(os.path.join(OUT,"partial_ot_tracks.vtp")); w.SetInputData(pd); w.Write()
write_points(); write_tracks()

# A PVD makes the original 21-field sequence explicitly time-dependent in ParaView.
with open(os.path.join(OUT,"cylinder_timeseries.pvd"),"w") as f:
  f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>\n')
  for s in range(1,22): f.write(f'<DataSet timestep="{s-1}" group="" part="0" file="../cylinder/cylinder{s}.vti"/>\n')
  f.write('</Collection></VTKFile>\n')

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()
field=PVDReader(FileName=os.path.join(OUT,"cylinder_timeseries.pvd")); field.PointArrays=["u","v"]
field.UpdatePipeline(20)
calc=Calculator(Input=field); calc.ResultArrayName="Velocity"; calc.Function="u*iHat + v*jHat"; calc.UpdatePipeline(20)
glyph=Glyph(Input=calc, GlyphType="Arrow"); glyph.OrientationArray=["POINTS","Velocity"]; glyph.ScaleArray=["POINTS","Velocity"]; glyph.ScaleFactor=35.0; glyph.GlyphMode="Every Nth Point"; glyph.Stride=18
crit=XMLPolyDataReader(FileName=[os.path.join(OUT,"critical_points.vtp")])
sphere=Glyph(Input=crit,GlyphType="Sphere"); sphere.ScaleFactor=4.0; sphere.GlyphMode="All Points"
tracks=XMLPolyDataReader(FileName=[os.path.join(OUT,"partial_ot_tracks.vtp")])
tube=Tube(Input=tracks); tube.Radius=1.1; tube.NumberofSides=12
view=GetActiveViewOrCreate('RenderView'); view.ViewSize=[1400,900]; view.Background=[0.08,0.09,0.13]
Show(field,view).Visibility=0
g=Show(glyph,view); ColorBy(g,("POINTS","Velocity","Magnitude")); g.SetScalarBarVisibility(view,True)
c=Show(sphere,view); ColorBy(c,("POINTS","TimeStep")); c.SetScalarBarVisibility(view,True)
tr=Show(tube,view); tr.DiffuseColor=[1.0,0.85,0.2]
view.ViewTime=20; view.InteractionMode='2D'
ResetCamera(view); view.CameraParallelProjection=1; view.CameraParallelScale=250
Render(); SaveScreenshot(os.path.join(OUT,"cylinder_partial_ot_tracking.png"),view)
SaveState(os.path.join(OUT,"cylinder_partial_ot_tracking.pvsm"))
print("wrote",OUT)
