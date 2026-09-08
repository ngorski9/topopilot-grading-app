from paraview.simple import *
r = XMLImageDataReader(FileName=['/workspace/Ionization/Ionization1.vti'])
s = TTKTopologicalSimplificationByPersistence(Input=r)
s.InputArray=['POINTS','Scalars_']; s.PersistenceThreshold=0.1; s.ThresholdIsAbsolute=1
t = TTKContourTree(Input=s)
t.ScalarField=['POINTS','Scalars_']
UpdatePipeline(proxy=t)
print('outputs', t.GetNumberOfOutputPorts())
for p in range(t.GetNumberOfOutputPorts()):
  d=t.GetClientSideObject().GetOutputDataObject(p)
  print('bounds', d.GetBounds())
  print('port',p,'class',d.GetClassName(),'pts',d.GetNumberOfPoints(),'cells',d.GetNumberOfCells(),'types', [d.GetCellType(i) for i in range(min(5,d.GetNumberOfCells()))])
  print(' point arrays',[d.GetPointData().GetArrayName(i) for i in range(d.GetPointData().GetNumberOfArrays())])
  print(' cell arrays',[d.GetCellData().GetArrayName(i) for i in range(d.GetCellData().GetNumberOfArrays())])
