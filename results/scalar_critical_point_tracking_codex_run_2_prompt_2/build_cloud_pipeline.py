from pathlib import Path

from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLImageDataWriter, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkFiltersCore import vtkThresholdPoints
from topologytoolkit import ttkTopologicalSimplificationByPersistence, ttkTrackingFromFields

ROOT = Path('/workspace')
FILES = [ROOT / 'cloud' / f'cloud{i}.vti' for i in (1, 2, 3)]
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

def read(path):
    r = vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()

fields = [read(p) for p in FILES]

# Save an explicitly named original field for convenient ParaView loading.
orig_writer = vtkXMLImageDataWriter()
orig_writer.SetFileName(str(OUT / 'cloud_original_t0.vti'))
orig_writer.SetInputData(fields[0])
orig_writer.Write()

# Persistence simplification of the original scalar field.  A threshold of 0.5
# is retained as a normalized fraction of the scalar range, matching TTK's UI.
simp = ttkTopologicalSimplificationByPersistence()
simp.SetInputData(fields[0])
simp.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'Scalars_')
simp.SetPersistenceThreshold(0.5)
simp.SetThresholdIsAbsolute(False)
simp.Update()
simp_writer = vtkXMLImageDataWriter()
simp_writer.SetFileName(str(OUT / 'cloud_persistence_simplified_threshold_0.5.vti'))
simp_writer.SetInputData(simp.GetOutput())
simp_writer.Write()

# TrackingFromFields takes successive time samples as point-data arrays on one grid.
track_input = fields[0].NewInstance()
track_input.DeepCopy(fields[0])
pd = track_input.GetPointData()
for i, field in enumerate(fields):
    a = field.GetPointData().GetArray('Scalars_')
    a2 = a.NewInstance()
    a2.DeepCopy(a)
    a2.SetName(f'Scalars_t{i}')
    pd.AddArray(a2)

tracker = ttkTrackingFromFields()
tracker.SetInputData(track_input)
tracker.SetStartTimestep(0)
tracker.SetEndTimestep(2)
tracker.SetSampling(1)
tracker.SetTolerance(0.0)
# Wasserstein-1 is the earth mover's distance.
tracker.SetWassersteinMetric('1')
tracker.SetDistanceAlgorithm('ttk')
tracker.SetDoPostProc(True)
tracker.Update()
tracking = tracker.GetOutput()

# Persist the exact TTK tracking mesh.  Its maxima trajectories are retained
# as the output geometry and colored by the TTK time-step attribute in ParaView.
tw = vtkXMLPolyDataWriter()
# Tracking output is normally an unstructured grid: use legacy VTK writer path
# in the generated state for full fidelity; report summary here.
from vtkmodules.vtkIOLegacy import vtkDataSetWriter
lw = vtkDataSetWriter()
lw.SetFileName(str(OUT / 'cloud_maxima_tracking_emd_3steps.vtk'))
lw.SetInputData(tracking)
lw.Write()

print('original range:', fields[0].GetPointData().GetArray('Scalars_').GetRange())
print('simplified arrays:', [simp.GetOutput().GetPointData().GetArrayName(i) for i in range(simp.GetOutput().GetPointData().GetNumberOfArrays())])
print('tracking points/cells:', tracking.GetNumberOfPoints(), tracking.GetNumberOfCells())
print('tracking arrays:', [tracking.GetPointData().GetArrayName(i) for i in range(tracking.GetPointData().GetNumberOfArrays())])
print('cell arrays:', [tracking.GetCellData().GetArrayName(i) for i in range(tracking.GetCellData().GetNumberOfArrays())])
