import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.optimize import root
from PIL import Image, ImageDraw
from paraview.simple import *

reader = vtk.vtkXMLImageDataReader(); reader.SetFileName('/workspace/Ocean.vti'); reader.Update()
data = reader.GetOutput()
A, B, C, D = [vtk_to_numpy(data.GetPointData().GetArray(n)).reshape(101,101) for n in 'ABCD']

# 10 samples per unit-square, with the major real eigenvector encoded as hue.
n = 1001; yy, xx = np.mgrid[0:n,0:n]; x, y = xx/10., yy/10.
i, j = np.minimum(x.astype(int),99), np.minimum(y.astype(int),99); u, v = x-i, y-j
def sample(f): return (1-u)*(1-v)*f[j,i]+u*(1-v)*f[j,i+1]+(1-u)*v*f[j+1,i]+u*v*f[j+1,i+1]
a,b,c,d = [sample(f) for f in (A,B,C,D)]
disc = (a-d)**2 + 4*b*c; r = np.sqrt(np.maximum(disc,0))
vx, vy = b.copy(), (d-a+r)/2
alt = (np.abs(vx)+np.abs(vy)) < 1e-12; vx[alt], vy[alt] = (a[alt]-d[alt]+r[alt])/2, c[alt]
angle = np.mod(np.arctan2(vy,vx),np.pi)/np.pi
from matplotlib.colors import hsv_to_rgb
rgb = hsv_to_rgb(np.dstack((angle,np.full_like(angle,.72),np.full_like(angle,.93))))
rgb[disc < 0] = (.13,.16,.20)

# Zeros of (A-D, B+C) are degenerate points.  Index sign gives wedge/trisector.
H,G=A-D,B+C; pts=[]
for J in range(100):
 for I in range(100):
  q=np.array([[H[J,I],G[J,I]],[H[J,I+1],G[J,I+1]],[H[J+1,I],G[J+1,I]],[H[J+1,I+1],G[J+1,I+1]]])
  if not(q[:,0].min()<=0<=q[:,0].max() and q[:,1].min()<=0<=q[:,1].max()): continue
  def f(z):
   U,V=z; return (1-U)*(1-V)*q[0]+U*(1-V)*q[1]+(1-U)*V*q[2]+U*V*q[3]
  U,V=root(f,[.5,.5]).x
  if not (0<=U<=1 and 0<=V<=1 and np.linalg.norm(f((U,V)))<1e-6): continue
  X,Y=I+U,J+V
  if any((X-X0)**2+(Y-Y0)**2<1e-4 for X0,Y0,_ in pts): continue
  hu=(1-V)*(q[1,0]-q[0,0])+V*(q[3,0]-q[2,0]); hv=(1-U)*(q[2,0]-q[0,0])+U*(q[3,0]-q[1,0])
  gu=(1-V)*(q[1,1]-q[0,1])+V*(q[3,1]-q[2,1]); gv=(1-U)*(q[2,1]-q[0,1])+U*(q[3,1]-q[1,1])
  pts.append((X,Y,'Wedge' if hu*gv-hv*gu>0 else 'Trisector'))

im=vtk.vtkImageData(); im.SetDimensions(n,n,1)
sc=numpy_to_vtk((rgb*255).astype(np.uint8).reshape(-1,3),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR); sc.SetNumberOfComponents(3); im.GetPointData().SetScalars(sc)
w=vtk.vtkPNGWriter(); w.SetFileName('/workspace/ocean_eigenvector_partition.png'); w.SetInputData(im); w.Write()
image=Image.open('/workspace/ocean_eigenvector_partition.png').convert('RGB'); draw=ImageDraw.Draw(image)
for X,Y,t in pts:
 cx,cy=round(X*10),round((100-Y)*10); col=(255,105,160) if t=='Trisector' else (255,255,255)
 draw.ellipse((cx-10,cy-10,cx+10,cy+10),fill=col,outline=(20,20,25),width=2)
image.save('/workspace/ocean_eigenvector_partition.png')

p=vtk.vtkPoints(); kinds=vtk.vtkStringArray(); kinds.SetName('DegenerateType'); cells=vtk.vtkCellArray()
for q,(X,Y,t) in enumerate(pts): p.InsertNextPoint(X,Y,0); kinds.InsertNextValue(t); cells.InsertNextCell(1); cells.InsertCellPoint(q)
out=vtk.vtkPolyData(); out.SetPoints(p); out.SetVerts(cells); out.GetPointData().AddArray(kinds)
pw=vtk.vtkXMLPolyDataWriter(); pw.SetFileName('/workspace/ocean_degenerate_points.vtp'); pw.SetInputData(out); pw.Write()

partition=PNGSeriesReader(FileNames=['/workspace/ocean_eigenvector_partition.png']); degenerate=XMLPolyDataReader(FileName=['/workspace/ocean_degenerate_points.vtp'])
view=GetActiveViewOrCreate('RenderView'); view.ViewSize=[1001,1001]; view.InteractionMode='2D'; Show(partition,view); Hide(degenerate,view); SaveState('/workspace/ocean_eigenvector_partition.pvsm')
print(len(pts), 'points:', sum(t=='Trisector' for _,_,t in pts),'trisectors;',sum(t=='Wedge' for _,_,t in pts),'wedges')
