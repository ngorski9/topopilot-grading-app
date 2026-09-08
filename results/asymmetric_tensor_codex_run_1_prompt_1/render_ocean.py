import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('Ocean.vti')
reader.Update()
image = reader.GetOutput()
nx, ny, _ = image.GetDimensions()
pd = image.GetPointData()
def field(name): return vtk_to_numpy(pd.GetArray(name)).reshape(ny, nx)
A, B, C, D = (field(n) for n in ('A', 'B', 'C', 'D'))

# The unoriented principal-eigenvector line field is represented by the
# symmetric traceless coordinates (u,v).  This is also well defined for the
# asymmetric input, while the antisymmetric part only shifts eigenvalues.
u = A - D
v = B + C

def poly_actor(points, lines, color, width=1.0):
    pts = vtk.vtkPoints()
    for p in points: pts.InsertNextPoint(*p)
    cells = vtk.vtkCellArray()
    for a, b in lines:
        cells.InsertNextCell(2); cells.InsertCellPoint(a); cells.InsertCellPoint(b)
    data = vtk.vtkPolyData(); data.SetPoints(pts); data.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(data)
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(width); actor.GetProperty().SetLighting(False)
    return actor

# 10 pixels per input square: a short principal-eigenvector segment per cell.
points=[]; lines=[]
for j in range(ny - 1):
    for i in range(nx - 1):
        uu=float((u[j,i]+u[j+1,i]+u[j,i+1]+u[j+1,i+1])/4)
        vv=float((v[j,i]+v[j+1,i]+v[j,i+1]+v[j+1,i+1])/4)
        theta=0.5*math.atan2(vv, uu)
        dx,dy=0.39*math.cos(theta),0.39*math.sin(theta)
        k=len(points); points.extend([(i+.5-dx,j+.5-dy,0.03),(i+.5+dx,j+.5+dy,0.03)]); lines.append((k,k+1))
vectors = poly_actor(points, lines, (0.20,0.78,0.92), 1.15)

# Locate simultaneous zeroes of (u,v) in each piecewise-linear triangle.
# The sign of det(grad(u), grad(v)) gives line-field index: +1/2 wedge,
# -1/2 trisector.
wedges=[]; trisectors=[]
for j in range(ny-1):
    for i in range(nx-1):
        for tri in ((0,1,2),(3,2,1)):
            # vertex ordering: bottom-left, bottom-right, top-left, top-right
            ids=((i,j),(i+1,j),(i,j+1),(i+1,j+1)); q=[ids[t] for t in tri]
            M=[]
            for x,y in q: M.append((float(u[y,x]),float(v[y,x])))
            du1,dv1=M[1][0]-M[0][0],M[1][1]-M[0][1]
            du2,dv2=M[2][0]-M[0][0],M[2][1]-M[0][1]
            det=du1*dv2-du2*dv1
            if abs(det) < 1e-12: continue
            # barycentric coordinates for u=v=0
            l1=(-M[0][0]*dv2 + M[0][1]*du2)/det
            l2=(-du1*M[0][1] + dv1*M[0][0])/det
            l0=1-l1-l2
            eps=1e-7
            if l0 >= -eps and l1 >= -eps and l2 >= -eps:
                x=l0*q[0][0]+l1*q[1][0]+l2*q[2][0]
                y=l0*q[0][1]+l1*q[1][1]+l2*q[2][1]
                (wedges if det > 0 else trisectors).append((x,y,0.08))

def spheres(locations, color):
    app=vtk.vtkAppendPolyData()
    for x,y,z in locations:
        s=vtk.vtkSphereSource(); s.SetCenter(x,y,z); s.SetRadius(1.0); s.SetThetaResolution(20); s.SetPhiResolution(20); s.Update(); app.AddInputData(s.GetOutput())
    app.Update(); m=vtk.vtkPolyDataMapper(); m.SetInputConnection(app.GetOutputPort())
    a=vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(*color); a.GetProperty().SetLighting(False); return a

# Background panel outlines the 100x100 tensor domain.
plane=vtk.vtkPlaneSource(); plane.SetOrigin(0,0,0); plane.SetPoint1(100,0,0); plane.SetPoint2(0,100,0); plane.Update()
pm=vtk.vtkPolyDataMapper(); pm.SetInputConnection(plane.GetOutputPort())
pa=vtk.vtkActor(); pa.SetMapper(pm); pa.GetProperty().SetColor(0.025,0.075,0.12); pa.GetProperty().SetLighting(False)

ren=vtk.vtkRenderer(); ren.SetBackground(0.015,0.02,0.035); ren.AddActor(pa); ren.AddActor(vectors)
ren.AddActor(spheres(wedges,(1,1,1))); ren.AddActor(spheres(trisectors,(1,0.30,0.62)))
win=vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(1000,1000); win.AddRenderer(ren)
ren.GetActiveCamera().SetPosition(50,50,150); ren.GetActiveCamera().SetFocalPoint(50,50,0); ren.GetActiveCamera().SetViewUp(0,1,0); ren.GetActiveCamera().ParallelProjectionOn(); ren.GetActiveCamera().SetParallelScale(50)
win.Render()
w=vtk.vtkWindowToImageFilter(); w.SetInput(win); w.SetInputBufferTypeToRGBA(); w.ReadFrontBufferOff(); w.Update()
writer=vtk.vtkPNGWriter(); writer.SetFileName('output/Ocean_eigenvector_partition.png'); writer.SetInputConnection(w.GetOutputPort()); writer.Write()
print(f'Wedges: {len(wedges)}  Trisectors: {len(trisectors)}')
