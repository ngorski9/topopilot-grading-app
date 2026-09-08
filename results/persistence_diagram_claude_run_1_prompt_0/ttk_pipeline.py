from paraview.simple import *
import numpy as np

LoadPlugin('TopologyToolKit', remote=False, ns=globals())

reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

# 1. Compute initial persistence diagram to drive simplification
pd0 = TTKPersistenceDiagram(Input=reader)
pd0.ScalarField = ['POINTS', 'Scalars_']
pd0.UpdatePipeline()

# 2. Threshold the diagram to keep pairs with persistence >= 0.04
thresh = Threshold(Input=pd0)
thresh.Scalars = ['CELLS', 'Persistence']
thresh.LowerThreshold = 0.04
thresh.UpperThreshold = 1e18
thresh.ThresholdMethod = 'Between'
thresh.UpdatePipeline()

# 3. Topological simplification using the thresholded critical pairs
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=thresh)
simplify.ScalarField = ['POINTS', 'Scalars_']
simplify.UpdatePipeline()

# 4. Recompute persistence diagram on the simplified field
pd1 = TTKPersistenceDiagram(Input=simplify)
pd1.ScalarField = ['POINTS', 'Scalars_']
pd1.UpdatePipeline()

# Save the simplified persistence diagram data
SaveData('/workspace/QMCPACK_persistence_diagram.csv', proxy=pd1)

# Extract point data for plotting: PairIdentifier, Persistence, Birth/Death coords, PairType
from paraview.servermanager import Fetch
pd_data = Fetch(pd1)

# pd1 output is a vtkUnstructuredGrid where points represent birth/death pairs
# CriticalType / PairType stored as cell data typically; let's inspect fields
pdi = pd1.GetCellDataInformation()
print("Cell arrays in persistence diagram:")
for i in range(pdi.GetNumberOfArrays()):
    print(" ", pdi.GetArray(i).GetName())

pdip = pd1.GetPointDataInformation()
print("Point arrays in persistence diagram:")
for i in range(pdip.GetNumberOfArrays()):
    print(" ", pdip.GetArray(i).GetName())
