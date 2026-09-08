from paraview.simple import *
from paraview import servermanager

reader = XMLImageDataReader(FileName=['QMCPACK.vti'])
simp = TTKTopologicalSimplificationByPersistence(Input=reader)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.04
simp.ThresholdIsAbsolute = 1
pd = TTKPersistenceDiagram(Input=simp)
pd.ScalarField = ['POINTS', 'Scalars_']
UpdatePipeline(proxy=pd)
ss = servermanager.Fetch(simp)
print('simp arrays', [(ss.GetPointData().GetArrayName(i), ss.GetPointData().GetArray(i).GetRange()) for i in range(ss.GetPointData().GetNumberOfArrays())])
obj = servermanager.Fetch(pd)
print('class', obj.GetClassName(), 'points',obj.GetNumberOfPoints(), 'cells',obj.GetNumberOfCells())
for a in (obj.GetPointData(), obj.GetCellData()):
 print('arrays', [(a.GetArrayName(i), a.GetArray(i).GetNumberOfComponents()) for i in range(a.GetNumberOfArrays())])
for i in range(min(obj.GetNumberOfPoints(), 10)): print('pt',i,obj.GetPoint(i))
for i in range(min(obj.GetNumberOfCells(), 10)):
 print('cell',i, obj.GetCellData().GetArray('PairType').GetTuple1(i) if obj.GetCellData().GetArray('PairType') else '?')
