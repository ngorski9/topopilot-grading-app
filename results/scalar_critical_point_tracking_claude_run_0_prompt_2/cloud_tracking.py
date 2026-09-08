from paraview.simple import *
import glob, os

paraview.simple._DisableFirstRenderCameraReset()

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# --- Import the cloud time-varying dataset (first 3 time steps for tracking) ---
files = sorted(glob.glob(os.path.join(WORKDIR, "cloud", "cloud*.vti")),
               key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))[:3]
print("Using time steps:", files)

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ViewSize = [1200, 900]

scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Warm to Cold', True)

# --- 1) Show the original scalar field ---
origDisplay = Show(reader, renderView1)
ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origDisplay.RescaleTransferFunctionToDataRange(True)
origDisplay.SetScalarBarVisibility(renderView1, True)
renderView1.ResetCamera()
SaveScreenshot(os.path.join(WORKDIR, "01_original_scalar_field.png"), renderView1)
Hide(reader, renderView1)

# --- 2) Persistence diagram + topological simplification (threshold = 0.5), one pipeline per time step ---
maximaDiagrams = []
simplifiedFields = []
for f in files:
    r = XMLImageDataReader(FileName=[f])
    r.PointArrayStatus = ['Scalars_']
    r.UpdatePipeline()

    pd = TTKPersistenceDiagram(Input=r)
    pd.ScalarField = ['POINTS', 'Scalars_']

    th = Threshold(Input=pd)
    th.Scalars = ['CELLS', 'Persistence']
    th.ThresholdMethod = 'Above Upper Threshold'
    th.UpperThreshold = 0.5

    simp = TTKTopologicalSimplification(Domain=r, Constraints=th)
    simp.ScalarField = ['POINTS', 'Scalars_']
    simp.UpdatePipeline()
    simplifiedFields.append(simp)

    simpPD = TTKPersistenceDiagram(Input=simp)
    simpPD.ScalarField = ['POINTS', 'Scalars_']

    # Keep only saddle-maximum pairs (PairType == 1): piecewise-linear maxima
    maxOnly = Threshold(Input=simpPD)
    maxOnly.Scalars = ['CELLS', 'PairType']
    maxOnly.ThresholdMethod = 'Between'
    maxOnly.LowerThreshold = 1
    maxOnly.UpperThreshold = 1
    maxOnly.UpdatePipeline()
    maximaDiagrams.append(maxOnly)

# Show the persistence-simplified scalar field (first time step) for the middle screenshot
simplifiedDisplay = Show(simplifiedFields[0], renderView1)
ColorBy(simplifiedDisplay, ('POINTS', 'Scalars_'))
simplifiedDisplay.RescaleTransferFunctionToDataRange(True)
simplifiedDisplay.SetScalarBarVisibility(renderView1, True)
renderView1.ResetCamera()
SaveScreenshot(os.path.join(WORKDIR, "02_persistence_simplified.png"), renderView1)
Hide(simplifiedFields[0], renderView1)

# --- 3) Track maxima across the 3 time steps using earth mover's distance (Wasserstein assignment) ---
tracking = TTKTrackingFromPersistenceDiagrams(Input=maximaDiagrams)
tracking.Persistencethreshold = 0.5
tracking.Assignmentmethod = 'ttk: pMunkres (Wasserstein), Gabow-Tarjan (Bottleneck)'
tracking.UpdatePipeline()

# --- 4) Visualize tracked critical points (spheres, radius 2) + scalar field, warm-cold colormap ---
Show(reader, renderView1)
ColorBy(GetDisplayProperties(reader, renderView1), ('POINTS', 'Scalars_'))
GetDisplayProperties(reader, renderView1).RescaleTransferFunctionToDataRange(True)
GetDisplayProperties(reader, renderView1).SetScalarBarVisibility(renderView1, True)

trackDisplay = Show(tracking, renderView1)
trackDisplay.Representation = 'Point Gaussian'
trackDisplay.GaussianRadius = 2.0
trackDisplay.ShaderPreset = 'Sphere'
ColorBy(trackDisplay, ('CELLS', 'Persistence'))
trackDisplay.RescaleTransferFunctionToDataRange(True)
persistenceLUT = GetColorTransferFunction('Persistence')
persistenceLUT.ApplyPreset('Warm to Cold', True)
trackDisplay.SetScalarBarVisibility(renderView1, True)

renderView1.ResetCamera()
SaveScreenshot(os.path.join(WORKDIR, "03_tracked_maxima.png"), renderView1)

print("Done. Screenshots written to", WORKDIR)
print(" - 01_original_scalar_field.png")
print(" - 02_persistence_simplified.png")
print(" - 03_tracked_maxima.png")
