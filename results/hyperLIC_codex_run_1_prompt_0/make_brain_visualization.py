from paraview.simple import *
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkIntArray

INPUT = '/workspace/brain.vti'
POINTS = '/workspace/brain_degenerate_points.vtp'
STATE = '/workspace/brain_degeneracies.pvsm'
IMAGE = '/workspace/brain_degeneracies.png'

r = vtkXMLImageDataReader()
r.SetFileName(INPUT)
r.Update()
img = r.GetOutput()
pd = img.GetPointData()
A, B, D = pd.GetArray('A'), pd.GetArray('B'), pd.GetArray('D')
dims, origin, spacing = img.GetDimensions(), img.GetOrigin(), img.GetSpacing()
nx, ny = dims[0], dims[1]

outpts = vtkPoints()
kind = vtkIntArray(); kind.SetName('DegeneracyType')
verts = vtkCellArray()
seen = set()

def val(arr, i, j): return arr.GetTuple1(i + nx*j)
def bilinear(q, u, v):
    return q[0]*(1-u)*(1-v) + q[1]*u*(1-v) + q[2]*(1-u)*v + q[3]*u*v
def deriv(q, u, v):
    return ((q[1]-q[0])*(1-v) + (q[3]-q[2])*v,
            (q[2]-q[0])*(1-u) + (q[3]-q[1])*u)

for j in range(ny-1):
    for i in range(nx-1):
        qa = [val(A,i,j)-val(D,i,j), val(A,i+1,j)-val(D,i+1,j),
              val(A,i,j+1)-val(D,i,j+1), val(A,i+1,j+1)-val(D,i+1,j+1)]
        qb = [2*val(B,i,j), 2*val(B,i+1,j), 2*val(B,i,j+1), 2*val(B,i+1,j+1)]
        u = v = 0.5
        for _ in range(20):
            f, g = bilinear(qa,u,v), bilinear(qb,u,v)
            au,av = deriv(qa,u,v); bu,bv = deriv(qb,u,v)
            det = au*bv-av*bu
            if abs(det) < 1.e-13: break
            du, dv = (f*bv-g*av)/det, (au*g-bu*f)/det
            u -= du; v -= dv
            if abs(du)+abs(dv) < 1.e-12: break
        f, g = bilinear(qa,u,v), bilinear(qb,u,v)
        if not (-1.e-8 <= u <= 1+1.e-8 and -1.e-8 <= v <= 1+1.e-8): continue
        if abs(f)+abs(g) > 1.e-7: continue
        au,av = deriv(qa,u,v); bu,bv = deriv(qb,u,v)
        jac = au*bv-av*bu
        if abs(jac) < 1.e-10: continue
        x, y = origin[0] + spacing[0]*(i+u), origin[1] + spacing[1]*(j+v)
        key = (round(x, 7), round(y, 7))
        if key in seen: continue
        seen.add(key)
        pid = outpts.InsertNextPoint(x,y,origin[2])
        verts.InsertNextCell(1); verts.InsertCellPoint(pid)
        kind.InsertNextValue(1 if jac > 0 else 0)

deg = vtkPolyData(); deg.SetPoints(outpts); deg.SetVerts(verts); deg.GetPointData().AddArray(kind)
w = vtkXMLPolyDataWriter(); w.SetFileName(POINTS); w.SetInputData(deg); w.Write()
print('degenerate points:', outpts.GetNumberOfPoints(), 'trisectors:', sum(kind.GetValue(k)==0 for k in range(kind.GetNumberOfTuples())), 'wedges:', sum(kind.GetValue(k)==1 for k in range(kind.GetNumberOfTuples())))

brain = XMLImageDataReader(registrationName='brain tensor field', FileName=[INPUT])
surface = ExtractSurface(registrationName='brain tensor field surface', Input=brain)
surfaceDisplay = Show(surface)
surfaceDisplay.Representation = 'Surface'
surfaceDisplay.ColorArrayName = ['POINTS', 'A']
surfaceDisplay.Opacity = 0.82
surfaceDisplay.SetScalarBarVisibility(GetActiveViewOrCreate('RenderView'), False)

degs = XMLPolyDataReader(registrationName='degenerate points', FileName=[POINTS])
trisectors = Threshold(registrationName='trisectors', Input=degs)
trisectors.Scalars = ['POINTS', 'DegeneracyType']; trisectors.LowerThreshold = 0; trisectors.UpperThreshold = 0
wedges = Threshold(registrationName='wedges', Input=degs)
wedges.Scalars = ['POINTS', 'DegeneracyType']; wedges.LowerThreshold = 1; wedges.UpperThreshold = 1
sphere = Sphere(Radius=1.0, ThetaResolution=20, PhiResolution=20)
triGlyphs = Glyph(registrationName='trisectors (radius 1)', Input=trisectors, GlyphType=sphere)
triGlyphs.OrientationArray = ['POINTS', 'No orientation array']; triGlyphs.ScaleArray = ['POINTS', 'No scale array']; triGlyphs.ScaleFactor = 1.0
wedgeGlyphs = Glyph(registrationName='wedges (radius 1)', Input=wedges, GlyphType=sphere)
wedgeGlyphs.OrientationArray = ['POINTS', 'No orientation array']; wedgeGlyphs.ScaleArray = ['POINTS', 'No scale array']; wedgeGlyphs.ScaleFactor = 1.0
triDisplay = Show(triGlyphs); triDisplay.DiffuseColor = [1.0, 0.35, 0.62]; triDisplay.AmbientColor = [1.0, 0.35, 0.62]; triDisplay.ColorArrayName = [None, '']
wedgeDisplay = Show(wedgeGlyphs); wedgeDisplay.DiffuseColor = [1.0, 1.0, 1.0]; wedgeDisplay.AmbientColor = [1.0, 1.0, 1.0]; wedgeDisplay.ColorArrayName = [None, '']

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 850]
view.Background = [0.08, 0.09, 0.12]
view.OrientationAxesVisibility = 0
view.InteractionMode = '2D'
view.CameraParallelProjection = 1
Render(); ResetCamera(view); view.CameraParallelScale *= 1.08; Render()
SaveScreenshot(IMAGE, view)
SaveState(STATE)

