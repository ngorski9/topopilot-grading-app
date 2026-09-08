from paraview.simple import *
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLImageDataWriter, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray, vtkIntArray
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np

INPUT = '/workspace/Ocean.vti'
PARTITION = '/workspace/Ocean_eigenvector_partition.vti'
DEGENERACIES = '/workspace/Ocean_degenerate_points.vtp'
STATE = '/workspace/Ocean_eigenvector_partition.pvsm'
SCREENSHOT = '/workspace/Ocean_eigenvector_partition.png'

# Read the four matrix components T=[[A,B],[C,D]].
reader = vtkXMLImageDataReader(); reader.SetFileName(INPUT); reader.Update()
src = reader.GetOutput(); dims = src.GetDimensions(); nx, ny = dims[0], dims[1]
pd = src.GetPointData()
def field(name):
    return vtk_to_numpy(pd.GetArray(name)).reshape((ny, nx))
A, B, C, D = (field(n) for n in ('A', 'B', 'C', 'D'))

# Ten pixels per unit square; samples are at output-pixel centres.
scale = 10; ox, oy = (nx - 1) * scale, (ny - 1) * scale
x = (np.arange(ox) + .5) / scale; y = (np.arange(oy) + .5) / scale
ix = np.minimum(x.astype(int), nx - 2); iy = np.minimum(y.astype(int), ny - 2)
tx = x - ix; ty = y - iy
def bilinear(f):
    f00=f[iy[:,None],ix[None,:]]; f10=f[iy[:,None],ix[None,:]+1]
    f01=f[iy[:,None]+1,ix[None,:]]; f11=f[iy[:,None]+1,ix[None,:]+1]
    return (1-ty)[:,None]*((1-tx)[None,:]*f00+tx[None,:]*f10) + ty[:,None]*((1-tx)[None,:]*f01+tx[None,:]*f11)
a,b,c,d = (bilinear(f) for f in (A,B,C,D))
gr = (c-b)/2.0
gs = np.sqrt((a-d)**2 + (b+c)**2)/2.0
# 0: CCW-complex, 1: CCW-real, 2: CW-real, 3: CW-complex.
labels = np.where(gr >= 0, np.where(np.abs(gr) > gs, 0, 1), np.where(np.abs(gr) > gs, 3, 2)).astype(np.uint8)

out = vtkImageData(); out.SetDimensions(ox, oy, 1); out.SetOrigin(.05,.05,0); out.SetSpacing(.1,.1,1)
arr = vtkUnsignedCharArray(); arr.SetName('EigenvectorPartition'); arr.SetNumberOfComponents(1); arr.SetNumberOfTuples(ox*oy)
arr.SetVoidArray(labels.ravel(), ox*oy, 1); out.GetPointData().SetScalars(arr)
w = vtkXMLImageDataWriter(); w.SetFileName(PARTITION); w.SetInputData(out); w.Write()

# Degenerate points of the dual eigenvector field satisfy A-D=0 and B+C=0.
# The input is PL, so solve the two affine equations in each triangle.
q=A-D; s=B+C; pts=[]; kinds=[]
for j in range(ny-1):
  for i in range(nx-1):
    for tri in ((0,1,3),(0,3,2)):
      xy=np.array([[i,j],[i+1,j],[i,j+1],[i+1,j+1]],float)[list(tri)]
      vv=np.array([[q[j,i],s[j,i]],[q[j,i+1],s[j,i+1]],
                   [q[j+1,i],s[j+1,i]],[q[j+1,i+1],s[j+1,i+1]]],float)[list(tri)]
      M=np.column_stack((np.ones(3),xy)); coef=np.linalg.solve(M,vv)
      try: p=np.linalg.solve(coef[1:].T,-coef[0])
      except np.linalg.LinAlgError: continue
      # barycentric inclusion with a small tolerance
      mat=np.column_stack((xy[1]-xy[0],xy[2]-xy[0]))
      try: uv=np.linalg.solve(mat,p-xy[0])
      except np.linalg.LinAlgError: continue
      if uv[0] >= -1e-8 and uv[1] >= -1e-8 and uv.sum() <= 1+1e-8:
        if not any(np.linalg.norm(p-z)<1e-5 for z in pts):
          # Positive determinant is a wedge; negative is a trisector.
          kinds.append(1 if np.linalg.det(coef[1:]) > 0 else 0)
          pts.append(p)

poly=vtkPolyData(); vpts=vtkPoints(); verts=vtkCellArray()
kindarr=vtkIntArray(); kindarr.SetName('DegenerateType'); kindarr.SetNumberOfComponents(1)
for p,k in zip(pts,kinds):
  pid=vpts.InsertNextPoint(float(p[0]),float(p[1]),0); verts.InsertNextCell(1); verts.InsertCellPoint(pid); kindarr.InsertNextValue(k)
poly.SetPoints(vpts); poly.SetVerts(verts); poly.GetPointData().AddArray(kindarr)
wp=vtkXMLPolyDataWriter(); wp.SetFileName(DEGENERACIES); wp.SetInputData(poly); wp.Write()

# Assemble a ParaView scene and save a reusable state + preview image.
Disconnect(); Connect()
partition = XMLImageDataReader(registrationName='Eigenvector partition (10 px/square)', FileName=[PARTITION])
deg = XMLPolyDataReader(registrationName='Degenerate points (radius 1)', FileName=[DEGENERACIES])
view=CreateView('RenderView'); view.ViewSize=[1200,1200]; view.InteractionMode='2D'; view.OrientationAxesVisibility=0
view.Background=[0.12,0.12,0.12]
pr=Show(partition,view); ColorBy(pr,('POINTS','EigenvectorPartition'))
lut=GetColorTransferFunction('EigenvectorPartition'); lut.InterpretValuesAsCategories=1; lut.Annotations=['0','CCW complex','1','CCW real','2','CW real','3','CW complex']
lut.IndexedColors=[0.90,0.31,0.25, 0.98,0.70,0.20, 0.18,0.58,0.78, 0.40,0.27,0.66]
pr.LookupTable=lut; pr.SetScalarBarVisibility(view,True)
sb=GetScalarBar(lut,view); sb.Title='Eigenvector partition'; sb.ComponentTitle=''; sb.LabelFontSize=14; sb.TitleFontSize=16
glyph=Glyph(registrationName='Degenerate points — radius 1', Input=deg, GlyphType='Sphere')
glyph.GlyphType.Radius=1.0; glyph.GlyphType.ThetaResolution=20; glyph.GlyphType.PhiResolution=20; glyph.ScaleArray=['POINTS','']; glyph.ScaleFactor=1.0
# Separate the two types so the requested colors are unambiguous.
tri=Threshold(registrationName='Trisectors (pink)',Input=glyph); tri.Scalars=['POINTS','DegenerateType']; tri.LowerThreshold=0; tri.UpperThreshold=0; tri.ThresholdMethod='Between'
wedge=Threshold(registrationName='Wedges (white)',Input=glyph); wedge.Scalars=['POINTS','DegenerateType']; wedge.LowerThreshold=1; wedge.UpperThreshold=1; wedge.ThresholdMethod='Between'
Hide(deg,view); Hide(glyph,view)
tr=Show(tri,view); tr.DiffuseColor=[1.0,0.30,0.60]; tr.AmbientColor=[1.0,0.30,0.60]; tr.Ambient=0.35
wr=Show(wedge,view); wr.DiffuseColor=[1,1,1]; wr.AmbientColor=[1,1,1]; wr.Ambient=0.35
ResetCamera(view); view.CameraParallelProjection=1; view.CameraParallelScale=55
Render(view); SaveScreenshot(SCREENSHOT,view,ImageResolution=[1200,1200]); SaveState(STATE)
print('degenerate points:',len(pts),'trisectors:',kinds.count(0),'wedges:',kinds.count(1))
