import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from paraview.simple import *

SOURCE = '/workspace/Ocean.vti'
PARTITION_FILE = '/workspace/Ocean_eigenvector_partition.vti'
POINTS_FILE = '/workspace/Ocean_degenerate_points.vtp'
STATE_FILE = '/workspace/Ocean_eigenvector_partition.pvsm'
IMAGE_FILE = '/workspace/Ocean_eigenvector_partition.png'

# Read the four (asymmetric) tensor components.
reader = vtk.vtkXMLImageDataReader(); reader.SetFileName(SOURCE); reader.Update()
src = reader.GetOutput(); nx, ny, _ = src.GetDimensions()
A, B, C, D = [vtk_to_numpy(src.GetPointData().GetArray(n)).reshape(ny, nx)
              for n in ('A', 'B', 'C', 'D')]

# A line-field angle for the real principal eigendirection.  The symmetric
# part determines its unoriented direction, and supplies a stable partition
# also where the nonsymmetric tensor is close to defective.
def angle(a, b, c, d):
    return 0.5 * np.arctan2(b + c, a - d) % np.pi

# The requested 10 pixels per input square: 1000 square pixels per side.
scale = 10
xx = np.arange((nx - 1) * scale + 1, dtype=float) / scale
yy = np.arange((ny - 1) * scale + 1, dtype=float) / scale
X, Y = np.meshgrid(xx, yy)
i = np.minimum(X.astype(int), nx - 2); j = np.minimum(Y.astype(int), ny - 2)
u = X - i; v = Y - j
def bilinear(F):
    return ((1-u)*(1-v)*F[j, i] + u*(1-v)*F[j, i+1]
            + (1-u)*v*F[j+1, i] + u*v*F[j+1, i+1])
T = angle(bilinear(A), bilinear(B), bilinear(C), bilinear(D))
# Twelve angular sectors make the eigenvector partition readily legible.
partition = np.floor(T / np.pi * 12).astype(np.uint8)

out = vtk.vtkImageData(); out.SetOrigin(0, 0, 0); out.SetSpacing(1/scale, 1/scale, 1)
out.SetDimensions(partition.shape[1], partition.shape[0], 1)
va = numpy_to_vtk(partition.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
va.SetName('EigenvectorPartition'); out.GetPointData().AddArray(va)
w = vtk.vtkXMLImageDataWriter(); w.SetFileName(PARTITION_FILE); w.SetInputData(out); w.Write()

# Locate zeroes of the symmetric traceless part in each triangle.  Their
# index is the sign of the local Jacobian: negative = trisector, positive = wedge.
P = vtk.vtkPoints(); kinds = vtk.vtkIntArray(); kinds.SetName('DegenerateType')
seen = set()
F1, F2 = A-D, B+C
for y in range(ny-1):
  for x in range(nx-1):
    corners = [(x,y),(x+1,y),(x+1,y+1),(x,y+1)]
    for tri in ((0,1,2),(0,2,3)):
      q = [corners[k] for k in tri]
      M = np.array([[F1[py,px] for px,py in q], [F2[py,px] for px,py in q], [1,1,1]], float)
      try: bary = np.linalg.solve(M, np.array([0.,0.,1.]))
      except np.linalg.LinAlgError: continue
      if np.all(bary >= -1e-9):
        px = sum(bary[k]*q[k][0] for k in range(3)); py = sum(bary[k]*q[k][1] for k in range(3))
        key=(round(px,6),round(py,6))
        if key in seen: continue
        seen.add(key)
        # Derivatives over this triangle, used for the wedge/trisector index.
        xy = np.array(q,float); vals=np.array([[F1[pyy,pxx],F2[pyy,pxx]] for pxx,pyy in q])
        G = np.column_stack((xy, np.ones(3)))
        coef=np.linalg.lstsq(G,vals,rcond=None)[0]
        det=np.linalg.det(coef[:2,:])
        P.InsertNextPoint(px,py,0); kinds.InsertNextValue(0 if det < 0 else 1)
poly=vtk.vtkPolyData(); poly.SetPoints(P); poly.GetPointData().AddArray(kinds)
verts=vtk.vtkCellArray()
for k in range(P.GetNumberOfPoints()): verts.InsertNextCell(1); verts.InsertCellPoint(k)
poly.SetVerts(verts)
pw=vtk.vtkXMLPolyDataWriter(); pw.SetFileName(POINTS_FILE); pw.SetInputData(poly); pw.Write()

# ParaView pipeline: partition + two independently styled point categories.
par = XMLImageDataReader(FileName=[PARTITION_FILE]); par.PointArrayStatus=['EigenvectorPartition']
tri = XMLPolyDataReader(FileName=[POINTS_FILE]); tri.PointArrayStatus=['DegenerateType']
trisectors = Threshold(Input=tri); trisectors.Scalars=['POINTS','DegenerateType']; trisectors.LowerThreshold=0; trisectors.UpperThreshold=0
wedges = Threshold(Input=tri); wedges.Scalars=['POINTS','DegenerateType']; wedges.LowerThreshold=1; wedges.UpperThreshold=1
view=CreateView('RenderView'); view.ViewSize=[1200,1000]; view.Background=[0.12,0.12,0.15]; view.InteractionMode='2D'; view.OrientationAxesVisibility=0
pd=Show(par,view); pd.Representation='Surface'; ColorBy(pd,('POINTS','EigenvectorPartition')); pd.LookupTable=GetColorTransferFunction('EigenvectorPartition')
lut=pd.LookupTable; lut.InterpretValuesAsCategories=1
lut.Annotations=sum(([str(k),str(k)] for k in range(12)), [])
lut.IndexedColors=[0.19,0.24,0.55, 0.20,0.49,0.75, 0.18,0.69,0.65, 0.38,0.80,0.45, 0.78,0.84,0.33, 0.98,0.70,0.22, 0.92,0.39,0.25, 0.72,0.20,0.43, 0.48,0.18,0.55, 0.28,0.25,0.65, 0.12,0.38,0.67, 0.10,0.50,0.70]
pd.SetScalarBarVisibility(view,False)
for obj, color in ((trisectors,[1.0,0.42,0.70]),(wedges,[1.0,1.0,1.0])):
  # Sphere glyphs give a geometric, data-space radius of one.
  g=Glyph(Input=obj, GlyphType='Sphere'); g.ScaleArray=['POINTS','No scale array']; g.ScaleFactor=1.0
  gd=Show(g,view); gd.DiffuseColor=color; gd.AmbientColor=color; gd.Specular=0.0
ResetCamera(view); view.CameraParallelProjection=1; view.CameraParallelScale=108
Render(view); SaveScreenshot(IMAGE_FILE,view)
SaveState(STATE_FILE)
print('partition:', PARTITION_FILE)
print('degenerate points:', P.GetNumberOfPoints(), 'trisectors:', sum(kinds.GetValue(i)==0 for i in range(kinds.GetNumberOfTuples())), 'wedges:', sum(kinds.GetValue(i)==1 for i in range(kinds.GetNumberOfTuples())))
