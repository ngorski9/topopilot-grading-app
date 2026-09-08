from paraview.simple import *
r=XMLUnstructuredGridReader(FileName=['/workspace/ion_tree_port_1.vtu'])
r.UpdatePipeline()
print(r.GetDataInformation().GetNumberOfPoints(), r.GetDataInformation().GetNumberOfCells())
x=TTKPlanarGraphLayout(Input=r)
print(x.ListProperties())
try:
 x.UpdatePipeline()
 print('out',x.GetDataInformation().GetNumberOfPoints(), x.GetDataInformation().GetNumberOfCells())
except Exception as e: print('ERR',e)
