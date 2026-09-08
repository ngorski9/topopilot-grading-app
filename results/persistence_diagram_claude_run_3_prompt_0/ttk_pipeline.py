from paraview.simple import *
import paraview

LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())

# 1. Load the scalar field
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()
pdi = reader.GetPointDataInformation()
array_name = pdi.GetArray(0).Name
print("Using scalar array:", array_name)

# Make sure it's the active scalar
reader.PointArrayStatus = [array_name]

# 2. Compute the Persistence Diagram (sublevel set filtration) on the ORIGINAL field
pd = TTKPersistenceDiagram(Input=reader)
pd.ScalarField = ['POINTS', array_name]
pd.UpdatePipeline()

# 3. Use TTKPersistenceCurve to help pick threshold, then Topological Simplification
#    using the persistence diagram to remove pairs below threshold=0.04
threshold = 0.04

persistence_range = pd.GetCellDataInformation().GetArray('Persistence').GetComponentRange(0)
max_persistence = persistence_range[1]

simplify_pairs = Threshold(Input=pd)
simplify_pairs.Scalars = ['CELLS', 'Persistence']
simplify_pairs.LowerThreshold = threshold
simplify_pairs.UpperThreshold = max_persistence * 1.1 + 1.0
simplify_pairs.ThresholdMethod = 'Between'
simplify_pairs.UpdatePipeline()
print("Pairs kept as constraints (persistence >=", threshold, "):", simplify_pairs.GetDataInformation().GetNumberOfPoints())

topo_simplify = TTKTopologicalSimplification(Domain=reader, Constraints=simplify_pairs)
topo_simplify.ScalarField = ['POINTS', array_name]
topo_simplify.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
topo_simplify.UpdatePipeline()

# 4. Recompute persistence diagram on the SIMPLIFIED field
simplified_array = array_name
pd_simplified = TTKPersistenceDiagram(Input=topo_simplify)
pd_simplified.ScalarField = ['POINTS', simplified_array]
pd_simplified.UpdatePipeline()

SaveData('/workspace/persistence_diagram_simplified.csv', proxy=pd_simplified)

# 5. Filter for pair orders (dimensions) 0 and 2 using the "PairType" cell field
#    PairType: 0 = min-saddle (order 0), 1 = saddle-saddle (order 1), 2 = saddle-max (order 2)
cdi = pd_simplified.GetCellDataInformation()
print("Cell data arrays:", [cdi.GetArray(i).Name for i in range(cdi.GetNumberOfArrays())])

t0 = Threshold(Input=pd_simplified)
t0.Scalars = ['CELLS', 'PairType']
t0.ThresholdMethod = 'Between'
t0.LowerThreshold = 0.0
t0.UpperThreshold = 0.0
t0.UpdatePipeline()

t2 = Threshold(Input=pd_simplified)
t2.Scalars = ['CELLS', 'PairType']
t2.ThresholdMethod = 'Between'
t2.LowerThreshold = 2.0
t2.UpperThreshold = 2.0
t2.UpdatePipeline()

merged = AppendDatasets(Input=[t0, t2])
merged.UpdatePipeline()

SaveData('/workspace/persistence_diagram_orders_0_2.csv', proxy=merged)

print("Done writing CSVs. Outputs written:")
print(" - /workspace/persistence_diagram_simplified.csv")
print(" - /workspace/persistence_diagram_orders_0_2.csv")
