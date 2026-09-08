from paraview.vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLImageDataReader

r=vtkXMLImageDataReader(); r.SetFileName('/workspace/Ocean.vti'); r.Update()
d=r.GetOutput(); pd=d.GetPointData()
print(d.GetDimensions(), d.GetBounds())
for n in ['A','B','C','D']:
 a=vtk_to_numpy(pd.GetArray(n)); print(n, a.min(),a.max())
a=vtk_to_numpy(pd.GetArray('A'));b=vtk_to_numpy(pd.GetArray('B'));c=vtk_to_numpy(pd.GetArray('C'));dd=vtk_to_numpy(pd.GetArray('D'))
q=(a-dd)**2+4*b*c
print('disc',q.min(),q.max(), (q<0).sum(), (abs(q)<1e-6).sum())
