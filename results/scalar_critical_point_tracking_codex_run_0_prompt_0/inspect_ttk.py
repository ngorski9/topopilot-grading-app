from paraview.simple import *
from paraview import servermanager
import os
LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())
for name in ['TTKPersistenceDiagram','TTKTopologicalSimplification','TTKTrackingFromFields','TTKIcospheresFromPoints']:
 print(name, name in globals())
 c=globals().get(name)
 if c:
  o=c()
  print(o.ListProperties())
  if name == 'TTKTrackingFromFields':
   p=o.GetProperty('Assignmentmethod')
   print('assignment',o.Assignmentmethod, p.GetAvailable())
