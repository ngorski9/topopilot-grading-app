import glob, re, os
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

LoadDistributedPlugin('TopologyToolKit', ns=globals())

base = '/workspace/cloud'
files = glob.glob(os.path.join(base, 'cloud*.vti'))
files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group()))

N_STEPS = 3
files = files[:N_STEPS]
print('Using files:', files)

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

tsteps = reader.TimestepValues
print('timesteps:', tsteps)

readers = []
for f in files:
    rr = XMLImageDataReader(FileName=[f])
    rr.PointArrayStatus = ['Scalars_']
    rr.UpdatePipeline()
    readers.append(rr)

grouped = GroupDatasets(Input=readers)
grouped.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ViewSize = [1000, 800]

scalarDisplay = Show(reader, renderView1, 'UniformGridRepresentation')
ColorBy(scalarDisplay, ('POINTS', 'Scalars_'))
scalarDisplay.RescaleTransferFunctionToDataRange(True)
scalarDisplay.SetScalarBarVisibility(renderView1, True)

lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Cool to Warm', True)

renderView1.ResetCamera()
renderView1.OrientationAxesVisibility = 0
SaveScreenshot('/workspace/cloud_scalarfield.png', renderView1, ImageResolution=[1000, 800])
print('Saved scalar field screenshot')

tracking = TTKTrackingFromFields(Input=grouped)
tracking.Persistencethreshold = 0.5
tracking.Firsttimestep = 0
tracking.Lasttimestep = N_STEPS - 1
tracking.Timesampling = 1
tracking.Assignmentmethod = 'ttk: sparse Munkres (Wasserstein), Gabow-Tarjan (Bottleneck)'
tracking.UpdatePipeline()

info = tracking.GetPointDataInformation()
for i in range(info.GetNumberOfArrays()):
    a = info.GetArrayInformation(i)
    print('Tracking output array:', a.GetName(), a.GetComponentRange(0))

maxima = Threshold(Input=tracking)
maxima.Scalars = ['POINTS', 'CriticalType']
maxima.ThresholdMethod = 'Between'
crit_range = tracking.GetPointDataInformation().GetArray('CriticalType').GetComponentRange(0)
max_type = crit_range[1]
maxima.LowerThreshold = max_type
maxima.UpperThreshold = max_type
maxima.UpdatePipeline()
print('Number of maxima points tracked:', maxima.GetDataInformation().GetNumberOfPoints())

spheres = TTKIcospheresFromPoints(Input=maxima)
spheres.Radius = 2.0
spheres.UpdatePipeline()

sphereDisplay = Show(spheres, renderView1)
ColorBy(sphereDisplay, ('POINTS', 'Scalars_'))
sphereDisplay.RescaleTransferFunctionToDataRange(True)

trackDisplay = Show(tracking, renderView1)
trackDisplay.SetRepresentationType('Surface')
ColorBy(trackDisplay, ('POINTS', 'Scalars_'))

renderView1.ResetCamera()
SaveScreenshot('/workspace/cloud_tracked_maxima.png', renderView1, ImageResolution=[1000, 800])
print('Saved tracked maxima screenshot')
print('DONE')
