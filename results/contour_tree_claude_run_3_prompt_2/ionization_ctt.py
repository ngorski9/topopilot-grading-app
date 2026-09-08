from paraview.simple import *

LoadPlugin('/opt/conda/lib/ttk-plugins/PersistenceDiagram.so', remote=False, ns=globals())
LoadPlugin('/opt/conda/lib/ttk-plugins/PersistenceCurve.so', remote=False, ns=globals())
LoadPlugin('/opt/conda/lib/ttk-plugins/TopologicalSimplification.so', remote=False, ns=globals())
LoadPlugin('/opt/conda/lib/ttk-plugins/FTMTree.so', remote=False, ns=globals())

reader = XMLPartitionedImageDataReader(FileName=[])
