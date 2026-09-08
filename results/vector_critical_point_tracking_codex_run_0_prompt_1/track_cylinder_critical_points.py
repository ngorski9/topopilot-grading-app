import os, csv, numpy as np, vtk, ot
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from paraview.simple import *
ROOT='/workspace'; OUT=ROOT+'/cylinder_tracking_output'; os.makedirs(OUT,exist_ok=True)
files=[ROOT+'/cylinder/cylinder%d.vti'%x for x in (1,11,21)]
def load(fn):
 r=vtk.vtkXMLImageDataReader(); r.SetFileName(fn); r.Update(); d=r.GetOutput(); nx,ny,_=d.GetDimensions()
 return vtk_to_numpy(d.GetPointData().GetArray('u')).reshape(ny,nx),vtk_to_numpy(d.GetPointData().GetArray('v')).reshape(ny,nx)
def crit(u,v):
 ny,nx=u.shape; out={}
 for j in range(ny-1):
  for i in range(nx-1):
   z=np.array([[u[j,i],v[j,i]],[u[j,i+1],v[j,i+1]],[u[j+1,i],v[j+1,i]],[u[j+1,i+1],v[j+1,i+1]]])
   for tri in ((0,1,3),(0,3,2)):
    q=z[list(tri)]; M=np.column_stack((q[1]-q[0],q[2]-q[0])); det=np.linalg.det(M)
    if abs(det)<1e-12: continue
    a,b=np.linalg.solve(M,-q[0])
    if a>=-1e-8 and b>=-1e-8 and a+b<=1.00000001:
     vs=np.array(((i,j),(i+1,j),(i,j+1),(i+1,j+1))); xy=(1-a-b)*vs[tri[0]]+a*vs[tri[1]]+b*vs[tri[2]]
     G=np.column_stack((vs[tri[1]]-vs[tri[0]],vs[tri[2]]-vs[tri[0]])); J=M@np.linalg.inv(G); ev=np.linalg.eigvals(J)
     k=0 if np.linalg.det(J)<0 else (1 if np.iscomplex(ev[0]) else 2)
     out[(round(float(xy[0]),7),round(float(xy[1]),7))]=k
 return np.array(list(out),float),np.array(list(out.values()),int)
fld=[load(x) for x in files]; cp=[crit(*x) for x in fld]; print('critical counts',*[len(x[0]) for x in cp])
tracks=[[(0,i,p,int(cp[0][1][i]))] for i,p in enumerate(cp[0][0])]
for st in range(2):
 a,b=cp[st][0],cp[st+1][0]; C=ot.dist(a,b,metric='sqeuclidean'); C/=max(C.max(),1.); mass=.85*min(len(a),len(b))/max(len(a),len(b)); g=ot.partial.partial_wasserstein(np.ones(len(a))/len(a),np.ones(len(b))/len(b),C,m=mass)
 useda=set(); usedb=set()
 for ia,ib in sorted(np.argwhere(g>1e-10),key=lambda z:g[tuple(z)],reverse=True):
  ia=int(ia); ib=int(ib)
  if ia in useda or ib in usedb: continue
  useda.add(ia); usedb.add(ib)
  if st==0: tid=ia
  else:
   found=[n for n,t in enumerate(tracks) if t[-1][0]==st and t[-1][1]==ia]
   if not found: continue
   tid=found[0]
  tracks[tid].append((st+1,ib,b[ib],int(cp[st+1][1][ib])))
pts=vtk.vtkPoints(); poly=vtk.vtkPolyData(); verts=vtk.vtkCellArray(); ts=[]; ks=[]; ids=[]
for tid,t in enumerate(tracks):
 for st,_,xy,k in t:
  p=pts.InsertNextPoint(float(xy[0]),float(xy[1]),1.5); c=vtk.vtkVertex(); c.GetPointIds().SetId(0,p); verts.InsertNextCell(c); ts.append(st);ks.append(k);ids.append(tid)
poly.SetPoints(pts);poly.SetVerts(verts)
for n,x in [('TimeStep',ts),('CriticalType',ks),('TrackId',ids)]: q=numpy_to_vtk(np.array(x),deep=True);q.SetName(n);poly.GetPointData().AddArray(q)
w=vtk.vtkXMLPolyDataWriter();w.SetFileName(OUT+'/tracked_critical_points.vtp');w.SetInputData(poly);w.Write()
lnpoly=vtk.vtkPolyData();lnpoly.SetPoints(pts); lines=vtk.vtkCellArray(); off=0
for t in tracks:
 if len(t)>1:
  ln=vtk.vtkPolyLine();ln.GetPointIds().SetNumberOfIds(len(t))
  for q in range(len(t)):ln.GetPointIds().SetId(q,off+q)
  lines.InsertNextCell(ln)
 off+=len(t)
lnpoly.SetLines(lines);w=vtk.vtkXMLPolyDataWriter();w.SetFileName(OUT+'/critical_point_tracks.vtp');w.SetInputData(lnpoly);w.Write()
with open(OUT+'/tracking_summary.csv','w',newline='') as f:
 wr=csv.writer(f);wr.writerow(['track_id','time_step','x','y','type'])
 for tid,t in enumerate(tracks):
  for st,_,xy,k in t: wr.writerow([tid,st,xy[0],xy[1],['saddle','focus','node'][k]])
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
u,v=fld[-1]; yy,xx=np.mgrid[0:u.shape[0]:12,0:u.shape[1]:12]
fig,ax=plt.subplots(figsize=(12,7)); ax.quiver(xx,yy,u[::12,::12],v[::12,::12],color='#6b7280',scale=35,width=.0018)
colors=['#2563eb','#16a34a','#dc2626']
for t in tracks:
 if len(t)>1: ax.plot([q[2][0] for q in t],[q[2][1] for q in t],color='#111827',lw=.75,alpha=.65,zorder=2)
for st,(p,k) in enumerate(cp): ax.scatter(p[:,0],p[:,1],s=28,c=colors[st],label='frame '+str((1,11,21)[st]),edgecolors='white',linewidths=.25,zorder=3)
ax.set(xlim=(0,u.shape[1]-1),ylim=(0,u.shape[0]-1),aspect='equal',xlabel='x',ylabel='y',title='Cylinder vector field with partial-OT critical-point tracks');ax.legend(loc='upper right');fig.tight_layout();fig.savefig(OUT+'/tracked_critical_points.png',dpi=160);plt.close(fig)
field=XMLImageDataReader(FileName=files[-1]); calc=Calculator(Input=field);calc.ResultArrayName='Velocity';calc.Function='u*iHat + v*jHat'; mask=MaskPoints(Input=calc);mask.OnRatio=125
glyph=Glyph(Input=mask,GlyphType='Arrow');glyph.OrientationArray=['POINTS','Velocity'];glyph.ScaleArray=['POINTS','Velocity'];glyph.ScaleFactor=4
cpr=XMLPolyDataReader(FileName=[OUT+'/tracked_critical_points.vtp']);tr=XMLPolyDataReader(FileName=[OUT+'/critical_point_tracks.vtp']);view=CreateView('RenderView');view.ViewSize=[1200,800];view.InteractionMode='2D';view.Background=[1,1,1]
Show(glyph,view).DiffuseColor=[.25,.25,.25];d=Show(tr,view);d.DiffuseColor=[.1,.1,.1];d.LineWidth=3;d=Show(cpr,view);d.Representation='Points';d.PointSize=10;ColorBy(d,('POINTS','TimeStep'));lut=GetColorTransferFunction('TimeStep');lut.RGBPoints=[0,.1,.3,.9,1,.1,.75,.2,2,.9,.15,.1]
view.CameraParallelProjection=1;view.ResetCamera();Render();SaveState(OUT+'/cylinder_partial_ot_tracking.pvsm');print('wrote',OUT)
