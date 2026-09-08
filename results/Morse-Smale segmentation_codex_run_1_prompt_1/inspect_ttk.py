from paraview.simple import *
from paraview import servermanager

src = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
src.PointArrayStatus=['Scalars_']
simp = TTKTopologicalSimplificationByPersistence(Input=src)
simp.InputArray=['POINTS','Scalars_']
simp.PersistenceThreshold=0.05
simp.ThresholdIsAbsolute=1
msc=TTKMorseSmaleComplex(Input=simp)
msc.ScalarField=['POINTS','Scalars_']
msc.ThresholdIsAbsolute=1
msc.MorseSmaleComplexSegmentation=1
msc.CriticalPoints=1
UpdatePipeline(proxy=msc)
print('ports',msc.GetNumberOfOutputPorts())
for p in range(7):
  try:
    d=servermanager.Fetch(OutputPort(msc,p))
    print('port',p, type(d).__name__,'points',d.GetNumberOfPoints(),'cells',d.GetNumberOfCells())
    for a in (d.GetPointData(),d.GetCellData()):
      print([(a.GetArrayName(i),a.GetArray(i).GetRange()) for i in range(a.GetNumberOfArrays())])
  except Exception as e: print('port',p,'ERR',e)
