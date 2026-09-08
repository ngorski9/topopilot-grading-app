import csv, json, os
import numpy as np
import ot
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray, vtkIntArray, vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter, vtkActor, vtkPolyDataMapper
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkIOImage import vtkPNGWriter

OUT='cylinder_tracking'; os.makedirs(OUT,exist_ok=True)
files=[f'cylinder/cylinder{i}.vti' for i in (1,2,3)]
def load(f):
 r=vtkXMLImageDataReader(); r.SetFileName(f); r.Update(); d=r.GetOutput(); nx,ny,_=d.GetDimensions()
 return d,vtk_to_numpy(d.GetPointData().GetArray('u')).reshape(ny,nx),vtk_to_numpy(d.GetPointData().GetArray('v')).reshape(ny,nx)
def critical(u,v):
 ny,nx=u.shape; ans=[]
 for y in range(ny-1):
  for x in range(nx-1):
   cc=[(x,y),(x+1,y),(x+1,y+1),(x,y+1)]
   for ids in ((0,1,2),(0,2,3)):
    xy=np.array([cc[k] for k in ids],float); fu=np.array([u[cc[k][1],cc[k][0]] for k in ids]); fv=np.array([v[cc[k][1],cc[k][0]] for k in ids]); A=np.column_stack((np.ones(3),xy))
    try: au=np.linalg.solve(A,fu); av=np.linalg.solve(A,fv); J=np.array([[au[1],au[2]],[av[1],av[2]]]); p=np.linalg.solve(J,-np.array([au[0],av[0]])); bary=np.linalg.solve(np.vstack((xy.T,np.ones(3))),np.array([p[0],p[1],1.]))
    except np.linalg.LinAlgError: continue
    if min(bary)>=-1e-8 and max(bary)<=1+1e-8:
     det=float(np.linalg.det(J)); tr=float(np.trace(J)); typ=0 if det<0 else (1 if tr>0 else (2 if tr<0 else 3)); ans.append((p[0],p[1],typ,det,tr))
 out=[]
 for q in ans:
  if not any((q[0]-r[0])**2+(q[1]-r[1])**2<1e-10 for r in out): out.append(q)
 return out
data=[]; cps=[]
for f in files:
 d,u,v=load(f); data.append((d,u,v)); cps.append(critical(u,v))
print('critical points per time step:',[len(x) for x in cps])
def match(a,b):
 if not a or not b:return []
 A=np.array([[q[0],q[1]] for q in a]); B=np.array([[q[0],q[1]] for q in b]); C=ot.dist(A,B,metric='euclidean')/450
 C+=(np.array([q[2] for q in a])[:,None]!=np.array([q[2] for q in b])[None,:])*2
 G=ot.partial.partial_wasserstein(np.ones(len(a))/len(a),np.ones(len(b))/len(b),C,m=.90); out=[]
 for i in range(len(a)):
  j=int(np.argmax(G[i]))
  if G[i,j]>1e-10 and C[i,j]<.12: out.append((i,j,float(G[i,j]),float(C[i,j])))
 return out
m01=match(cps[0],cps[1]); m12=match(cps[1],cps[2]); print('partial-OT matches:',len(m01),len(m12))
n01={i:j for i,j,_,_ in m01}; n12={i:j for i,j,_,_ in m12}; tracks=[]; used=[set(),set(),set()]
for i in range(len(cps[0])):
 tr=[(0,i)]; used[0].add(i)
 if i in n01:
  j=n01[i]; tr.append((1,j)); used[1].add(j)
  if j in n12: tr.append((2,n12[j])); used[2].add(n12[j])
 tracks.append(tr)
for t in (1,2):
 for i in range(len(cps[t])):
  if i not in used[t]:tracks.append([(t,i)])
pts=vtkPoints(); verts=vtkCellArray(); ti=vtkIntArray(); ti.SetName('TimeStep'); tid=vtkIntArray(); tid.SetName('TrackId'); ct=vtkUnsignedCharArray(); ct.SetName('CriticalType')
for k,tr in enumerate(tracks):
 for t,i in tr:
  q=cps[t][i]; p=pts.InsertNextPoint(q[0],q[1],0); verts.InsertNextCell(1); verts.InsertCellPoint(p); ti.InsertNextValue(t+1); tid.InsertNextValue(k); ct.InsertNextValue(q[2])
pd=vtkPolyData(); pd.SetPoints(pts); pd.SetVerts(verts); pd.GetPointData().AddArray(ti); pd.GetPointData().AddArray(tid); pd.GetPointData().AddArray(ct)
w=vtkXMLPolyDataWriter(); w.SetFileName(OUT+'/tracked_critical_points.vtp'); w.SetInputData(pd); w.Write()
with open(OUT+'/partial_ot_matches.json','w') as f:json.dump({'step1_to_step2':m01,'step2_to_step3':m12,'tracks':tracks},f,indent=2)
with open(OUT+'/critical_points.csv','w',newline='') as f:
 z=csv.writer(f); z.writerow(['track_id','time_step','x','y','type','determinant','trace'])
 for k,tr in enumerate(tracks):
  for t,i in tr:z.writerow([k,t+1,*cps[t][i]])
ren=vtkRenderer(); ren.SetBackground(.97,.97,.97); win=vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(1650,700); colors=[(230,70,45),(40,130,220),(45,175,85),(150,80,180)]
for t,(d,u,v) in enumerate(data):
 nx,ny,_=d.GetDimensions(); off=t*170; p=vtkPoints(); vec=vtkFloatArray(); vec.SetNumberOfComponents(3); vec.SetName('Velocity'); vs=vtkCellArray()
 for y in range(4,ny,14):
  for x in range(4,nx,14):
   pid=p.InsertNextPoint(x+off,y,0); vs.InsertNextCell(1); vs.InsertCellPoint(pid); vec.InsertNextTuple3(float(u[y,x]),float(v[y,x]),0)
 g=vtkPolyData(); g.SetPoints(p); g.SetVerts(vs); g.GetPointData().SetVectors(vec); gl=vtkGlyph3D(); gl.SetInputData(g); ar=vtkArrowSource(); gl.SetSourceConnection(ar.GetOutputPort()); gl.SetVectorModeToUseVector(); gl.SetScaleModeToScaleByVector(); gl.SetScaleFactor(16); gl.OrientOn(); gl.Update(); mp=vtkPolyDataMapper(); mp.SetInputConnection(gl.GetOutputPort()); ac=vtkActor(); ac.SetMapper(mp); ac.GetProperty().SetColor(.25,.25,.25); ren.AddActor(ac)
 p=vtkPoints(); vs=vtkCellArray(); col=vtkUnsignedCharArray(); col.SetName('Color'); col.SetNumberOfComponents(3)
 for q in cps[t]:
  pid=p.InsertNextPoint(q[0]+off,q[1],.2); vs.InsertNextCell(1); vs.InsertCellPoint(pid); col.InsertNextTuple3(*colors[q[2]])
 g=vtkPolyData(); g.SetPoints(p); g.SetVerts(vs); g.GetPointData().SetScalars(col); mp=vtkPolyDataMapper(); mp.SetInputData(g); mp.SetScalarModeToUsePointData(); mp.SetColorModeToDirectScalars(); ac=vtkActor(); ac.SetMapper(mp); ac.GetProperty().SetPointSize(12); ren.AddActor(ac)
# Partial-OT trajectories connect corresponding critical points across the
# three displaced time panels.  Singletons intentionally have no line.
tp=vtkPoints(); lines=vtkCellArray()
for tr in tracks:
 if len(tr)>1:
  lines.InsertNextCell(len(tr))
  for t,i in tr:
   q=cps[t][i]; pid=tp.InsertNextPoint(q[0]+t*170,q[1],.1); lines.InsertCellPoint(pid)
tg=vtkPolyData(); tg.SetPoints(tp); tg.SetLines(lines); tm=vtkPolyDataMapper(); tm.SetInputData(tg); ta=vtkActor(); ta.SetMapper(tm); ta.GetProperty().SetColor(.12,.12,.12); ta.GetProperty().SetLineWidth(1.5); ta.GetProperty().SetOpacity(.65); ren.AddActor(ta)
ren.GetActiveCamera().ParallelProjectionOn(); ren.ResetCamera(); im=vtkWindowToImageFilter(); im.SetInput(win); im.Update(); pw=vtkPNGWriter(); pw.SetFileName(OUT+'/cylinder_partial_ot_tracking.png'); pw.SetInputConnection(im.GetOutputPort()); pw.Write(); print('wrote',OUT)
