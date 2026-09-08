from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkImageData
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter, vtkPolyDataMapper, vtkActor, vtkImageActor
from vtkmodules.vtkRenderingOpenGL2 import *
from vtkmodules.vtkIOImage import vtkPNGWriter
import math

reader = vtkXMLImageDataReader()
reader.SetFileName('brain.vti')
reader.Update()
img = reader.GetOutput()
pd = img.GetPointData()
A, B, D = pd.GetArray('A'), pd.GetArray('B'), pd.GetArray('D')
nx, ny, _ = img.GetDimensions()
origin, spacing = img.GetOrigin(), img.GetSpacing()

def ijpt(i, j): return (origin[0]+i*spacing[0], origin[1]+j*spacing[1], 0.)
def vals(i,j):
    k=j*nx+i
    return A.GetTuple1(k)-D.GetTuple1(k), 2.*B.GetTuple1(k)

# Locate zeros exactly within the two affine triangles used for each image cell.
# The sign of det(d(A-D,2B)/d(x,y)) is the eigenline singularity index:
# positive = wedge, negative = trisector.
roots=[]
eps=1e-8
for j in range(ny-1):
  for i in range(nx-1):
    for tri in (((i,j),(i+1,j),(i+1,j+1)), ((i,j),(i+1,j+1),(i,j+1))):
      p0,p1,p2=tri
      q0,q1,q2=vals(*p0),vals(*p1),vals(*p2)
      m00,m01=q1[0]-q0[0], q2[0]-q0[0]
      m10,m11=q1[1]-q0[1], q2[1]-q0[1]
      det=m00*m11-m01*m10
      if abs(det)<eps: continue
      u=((-q0[0])*m11 - m01*(-q0[1]))/det
      v=(m00*(-q0[1]) - (-q0[0])*m10)/det
      if u >= -eps and v >= -eps and u+v <= 1+eps:
        x=p0[0]+u*(p1[0]-p0[0])+v*(p2[0]-p0[0])
        y=p0[1]+u*(p1[1]-p0[1])+v*(p2[1]-p0[1])
        # remove coincident roots on a shared diagonal/edge
        if not any((x-r[0])**2+(y-r[1])**2 < 1e-10 for r in roots):
          roots.append((x,y, det>0))

# A muted anisotropy image provides the tensor-field context.
bg=vtkImageData(); bg.SetDimensions(nx,ny,1); bg.SetOrigin(origin); bg.SetSpacing(spacing)
rgba=vtkUnsignedCharArray(); rgba.SetName('RGBA'); rgba.SetNumberOfComponents(4); rgba.SetNumberOfTuples(nx*ny)
for j in range(ny):
  for i in range(nx):
    a,b=vals(i,j); mag=math.sqrt(a*a+b*b)
    # transparent outside tensor support, indigo within it
    alpha=0 if mag < 1e-7 else min(230, int(45+185*math.sqrt(mag/.000586)))
    rgba.SetTuple4(j*nx+i, 25, 58, 103, alpha)
bg.GetPointData().SetScalars(rgba)

# Principal-eigenvector strokes, sampled uniformly over the nonzero field.
pts=vtkPoints(); lines=vtkCellArray()
for j in range(2,ny-2,4):
  for i in range(2,nx-2,4):
    aa,bb,dd=A.GetTuple1(j*nx+i),B.GetTuple1(j*nx+i),D.GetTuple1(j*nx+i)
    an=math.hypot(aa-dd,2*bb)
    if an < 1e-6: continue
    theta=.5*math.atan2(2*bb,aa-dd)
    # keep glyph length restrained but encode anisotropy modestly
    half=min(1.55, .55+1.0*math.sqrt(an/.000586))
    x,y=ijpt(i,j)[:2]; dx,dy=half*math.cos(theta),half*math.sin(theta)
    n=pts.InsertNextPoint(x-dx,y-dy,0.08); m=pts.InsertNextPoint(x+dx,y+dy,0.08)
    lines.InsertNextCell(2); lines.InsertCellPoint(n); lines.InsertCellPoint(m)
field=vtkPolyData(); field.SetPoints(pts); field.SetLines(lines)
tube=vtkTubeFilter(); tube.SetInputData(field); tube.SetRadius(.12); tube.SetNumberOfSides(8); tube.Update()

def actor_for(poly, color):
  mapper=vtkPolyDataMapper(); mapper.SetInputData(poly)
  act=vtkActor(); act.SetMapper(mapper); act.GetProperty().SetColor(*color); return act
field_actor=actor_for(tube.GetOutput(), (0.58,0.82,1.0)); field_actor.GetProperty().SetOpacity(.9)

def point_actor(items, color):
  p=vtkPoints()
  for x,y,_ in items: p.InsertNextPoint(origin[0]+x*spacing[0],origin[1]+y*spacing[1],.45)
  poly=vtkPolyData(); poly.SetPoints(p)
  sph=vtkSphereSource(); sph.SetRadius(1.0); sph.SetThetaResolution(24); sph.SetPhiResolution(16)
  # Explicitly append transformed spheres so radius is in physical units.
  from vtkmodules.vtkFiltersCore import vtkGlyph3D
  glyph=vtkGlyph3D(); glyph.SetInputData(poly); glyph.SetSourceConnection(sph.GetOutputPort()); glyph.Update()
  act=actor_for(glyph.GetOutput(),color); act.GetProperty().SetSpecular(.45); act.GetProperty().SetSpecularPower(18); return act

wedges=[r for r in roots if r[2]]; tris=[r for r in roots if not r[2]]
renderer=vtkRenderer(); renderer.SetBackground(.025,.035,.06)
image_actor=vtkImageActor(); image_actor.GetMapper().SetInputData(bg); renderer.AddActor(image_actor)
renderer.AddActor(field_actor); renderer.AddActor(point_actor(wedges,(1,1,1))); renderer.AddActor(point_actor(tris,(1,.25,.62)))
window=vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1800,1100); window.AddRenderer(renderer)
cam=renderer.GetActiveCamera(); cam.ParallelProjectionOn(); cam.SetFocalPoint((nx-1)/2,(ny-1)/2,0); cam.SetPosition((nx-1)/2,(ny-1)/2,150); cam.SetParallelScale(ny+7)
renderer.ResetCameraClippingRange(); window.Render()
capture=vtkWindowToImageFilter(); capture.SetInput(window); capture.SetInputBufferTypeToRGBA(); capture.ReadFrontBufferOff(); capture.Update()
writer=vtkPNGWriter(); writer.SetFileName('brain_degenerate_points.png'); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()

# Save detected points for reproducibility.
out=vtkPolyData(); op=vtkPoints(); kind=vtkUnsignedCharArray(); kind.SetName('type'); kind.SetNumberOfComponents(1)
for x,y,w in roots: op.InsertNextPoint(x,y,0); kind.InsertNextValue(1 if w else 0)
out.SetPoints(op); out.GetPointData().AddArray(kind)
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
w=vtkXMLPolyDataWriter(); w.SetFileName('brain_degenerate_points.vtp'); w.SetInputData(out); w.Write()
print(f'Detected {len(wedges)} wedges and {len(tris)} trisectors; wrote brain_degenerate_points.png')
