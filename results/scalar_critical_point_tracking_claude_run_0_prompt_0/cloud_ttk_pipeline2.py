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

# ---- Visualize the original scalar field (first timestep) ----
reader0 = XMLImageDataReader(FileName=[files[0]])
reader0.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ViewSize = [1000, 800]

scalarDisplay = Show(reader0, renderView1)
ColorBy(scalarDisplay, ('POINTS', 'Scalars_'))
scalarDisplay.RescaleTransferFunctionToDataRange(True)
scalarDisplay.SetScalarBarVisibility(renderView1, True)

lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Cool to Warm', True)

renderView1.ResetCamera()
renderView1.OrientationAxesVisibility = 0
SaveScreenshot('/workspace/cloud_scalarfield.png', renderView1, ImageResolution=[1000, 800])
print('Saved scalar field screenshot')

# ---- Per-timestep: PL persistence diagram -> persistence simplification (threshold 0.5) -> keep maxima pairs ----
PERSISTENCE_THRESHOLD = 0.5
simplified_diagrams = []
for f in files:
    rr = XMLImageDataReader(FileName=[f])
    rr.UpdatePipeline()

    pd = TTKPersistenceDiagram(Input=rr)
    pd.ScalarField = ['POINTS', 'Scalars_']
    pd.UpdatePipeline()

    # persistence simplification: keep pairs with persistence above threshold
    persTh = Threshold(Input=pd)
    persTh.Scalars = ['CELLS', 'Persistence']
    persTh.ThresholdMethod = 'Above Upper Threshold'
    persTh.LowerThreshold = PERSISTENCE_THRESHOLD
    persTh.UpperThreshold = PERSISTENCE_THRESHOLD
    persTh.UpdatePipeline()

    # keep only saddle-maximum pairs (PairType == 1) -> tracks maxima
    maxTh = Threshold(Input=persTh)
    maxTh.Scalars = ['CELLS', 'PairType']
    maxTh.ThresholdMethod = 'Between'
    maxTh.LowerThreshold = 1
    maxTh.UpperThreshold = 1
    maxTh.UpdatePipeline()

    print(f, 'simplified maxima pairs:', maxTh.GetDataInformation().GetNumberOfCells())
    simplified_diagrams.append(maxTh)

grouped = GroupDatasets(Input=simplified_diagrams)
grouped.UpdatePipeline()

# ---- Track maxima across the 3 timesteps using earth mover's distance (Wasserstein) ----
tracking = TTKTrackingFromPersistenceDiagrams(Input=grouped)
tracking.Persistencethreshold = PERSISTENCE_THRESHOLD
tracking.Assignmentmethod = 'ttk: pMunkres (Wasserstein), Gabow-Tarjan (Bottleneck)'
tracking.UpdatePipeline()

print('Tracking output #points:', tracking.GetDataInformation().GetNumberOfPoints())
print('Tracking output #cells :', tracking.GetDataInformation().GetNumberOfCells())

# ---- Display tracked critical points (as spheres, radius 2) together with the scalar field ----
spheres = TTKIcospheresFromPoints(Input=tracking)
spheres.Radius = 2.0
spheres.UpdatePipeline()

sphereDisplay = Show(spheres, renderView1)
ColorBy(sphereDisplay, ('POINTS', 'Scalars'))
sphereDisplay.RescaleTransferFunctionToDataRange(True)
lut2 = GetColorTransferFunction('Scalars')
lut2.ApplyPreset('Cool to Warm', True)

trackDisplay = Show(tracking, renderView1)
trackDisplay.SetRepresentationType('Wireframe')
trackDisplay.LineWidth = 3.0
ColorBy(trackDisplay, ('POINTS', 'Scalars'))

renderView1.ResetCamera()
SaveScreenshot('/workspace/cloud_tracked_maxima.png', renderView1, ImageResolution=[1000, 800])
print('Saved tracked maxima screenshot')
print('DONE')
