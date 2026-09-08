from pathlib import Path

from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLImageDataWriter, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkPolyData

root = Path(__file__).resolve().parent
reader = vtkXMLImageDataReader()
reader.SetFileName(str(root / "brain.vti"))
reader.Update()
image = reader.GetOutput()
pd = image.GetPointData()
a, b, d = (pd.GetArray(n) for n in ("A", "B", "D"))

# Add the 3x3 embedding of the 2D symmetric tensor for ParaView's TensorGlyph.
tensors = vtkDoubleArray()
tensors.SetName("Tensor")
tensors.SetNumberOfComponents(9)
tensors.SetNumberOfTuples(image.GetNumberOfPoints())
for i in range(image.GetNumberOfPoints()):
    aa, bb, dd = a.GetTuple1(i), b.GetTuple1(i), d.GetTuple1(i)
    tensors.SetTuple9(i, aa, bb, 0., bb, dd, 0., 0., 0., 0.)
pd.AddArray(tensors)
writer = vtkXMLImageDataWriter()
writer.SetFileName(str(root / "brain_tensor.vti"))
writer.SetInputData(image)
writer.Write()

extent = image.GetExtent()
nx = extent[1] - extent[0] + 1
ny = extent[3] - extent[2] + 1
origin, spacing = image.GetOrigin(), image.GetSpacing()

def pid(ix, iy): return ix + iy * nx
def xy(ix, iy): return (origin[0] + ix * spacing[0], origin[1] + iy * spacing[1], 0.)

# In each PL triangle, solve [A-D, 2B] = [0, 0] in barycentric coordinates.
# Positive/negative Jacobian orientation corresponds to a wedge/trisector.
out = {"wedge": vtkPoints(), "trisector": vtkPoints()}
for j in range(ny - 1):
    for i in range(nx - 1):
        for tri in ((pid(i,j), pid(i+1,j), pid(i+1,j+1)),
                    (pid(i,j), pid(i+1,j+1), pid(i,j+1))):
            f = [a.GetTuple1(q) - d.GetTuple1(q) for q in tri]
            g = [2. * b.GetTuple1(q) for q in tri]
            m00, m01 = f[1]-f[0], f[2]-f[0]
            m10, m11 = g[1]-g[0], g[2]-g[0]
            det = m00*m11 - m01*m10
            if abs(det) < 1e-14:
                continue
            u = ((-f[0])*m11 - m01*(-g[0])) / det
            v = (m00*(-g[0]) - (-f[0])*m10) / det
            w = 1. - u - v
            # Half-open ownership prevents duplicate vertices along triangle edges.
            if u >= -1e-10 and v >= -1e-10 and w >= -1e-10:
                coords = [image.GetPoint(q) for q in tri]
                p = tuple(w*coords[0][k] + u*coords[1][k] + v*coords[2][k] for k in range(3))
                out["wedge" if det > 0 else "trisector"].InsertNextPoint(p)

for kind, points in out.items():
    poly = vtkPolyData(); poly.SetPoints(points)
    ids = vtkIntArray(); ids.SetName("Type"); ids.SetNumberOfTuples(points.GetNumberOfPoints())
    ids.FillComponent(0, 1 if kind == "wedge" else -1)
    poly.GetPointData().AddArray(ids)
    w = vtkXMLPolyDataWriter(); w.SetFileName(str(root / f"brain_{kind}s.vtp")); w.SetInputData(poly); w.Write()
    print(f"{kind}s: {points.GetNumberOfPoints()}")
