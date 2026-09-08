from paraview.simple import *
from paraview import servermanager

reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
simplified = TTKTopologicalSimplificationByPersistence(Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 0
diagram = TTKPersistenceDiagram(Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']
diagram.Dimensions = 'All Dimensions'
UpdatePipeline(proxy=diagram)
data = servermanager.Fetch(diagram)
print('OUT', data.GetClassName(), data.GetNumberOfPoints(), data.GetNumberOfCells())
pd=data.GetPointData(); cd=data.GetCellData()
for label, attrs in [('POINT',pd),('CELL',cd)]:
 print(label,attrs.GetNumberOfArrays())
 for i in range(attrs.GetNumberOfArrays()):
  a=attrs.GetArray(i); print(a.GetName(), a.GetRange())
for i in range(min(10,data.GetNumberOfCells())):
 c=data.GetCell(i); print('cell',i,'type',c.GetCellType(),'pairtype', cd.GetArray('PairType').GetTuple1(i) if cd.GetArray('PairType') else None)
