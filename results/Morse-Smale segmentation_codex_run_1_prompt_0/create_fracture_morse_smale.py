from paraview.simple import *

# Source and persistence-based scalar-field simplification.
fracture = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.05)', Input=fracture)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 1

# The fourth output is the piecewise-linear Morse--Smale manifold labeling.
msc = TTKMorseSmaleComplex(registrationName='Morse-Smale complex', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1

segmentation = ExtractSurface(registrationName='Piecewise-linear Morse-Smale segmentation', Input=OutputPort(msc, 3))

# For this 2-D domain, CellDimension is 0=minima, 1=saddles, 2=maxima.
def critical_subset(name, low, high, color):
    subset = Threshold(registrationName=name, Input=OutputPort(msc, 0))
    subset.Scalars = ['POINTS', 'CellDimension']
    subset.LowerThreshold = low
    subset.UpperThreshold = high
    spheres = TTKIcospheresFromPoints(registrationName=name + ' (radius 2)', Input=subset)
    spheres.Radius = 2.0
    spheres.Subdivisions = 3
    return spheres, color

minima, min_color = critical_subset('Minima', 0, 0, [0.0, 0.25, 1.0])
saddles, saddle_color = critical_subset('Saddles', 1, 1, [1.0, 1.0, 1.0])
maxima, max_color = critical_subset('Maxima', 2, 2, [1.0, 0.0, 0.0])

view = CreateView('RenderView')
view.ViewSize = [1600, 900]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
lut = GetColorTransferFunction('MorseSmaleManifold')
lut.InterpretValuesAsCategories = 1
# 28 manifold labels remain after the requested persistence simplification.
lut.Annotations = sum(([str(i), str(i)] for i in range(28)), [])
lut.IndexedColors = [
    0.23, 0.55, 0.75,  0.90, 0.55, 0.20,  0.35, 0.72, 0.38,
    0.70, 0.35, 0.65,  0.85, 0.80, 0.25,  0.35, 0.70, 0.70,
    0.85, 0.42, 0.35,  0.50, 0.50, 0.82,
    0.18, 0.45, 0.62,  0.95, 0.65, 0.25,  0.25, 0.65, 0.30,
    0.72, 0.30, 0.62,  0.75, 0.70, 0.15,  0.25, 0.62, 0.65,
    0.80, 0.35, 0.28,  0.42, 0.42, 0.75,
    0.15, 0.60, 0.85,  0.98, 0.45, 0.18,  0.20, 0.80, 0.45,
    0.62, 0.25, 0.78,  0.92, 0.85, 0.20,  0.28, 0.82, 0.78,
    0.95, 0.30, 0.42,  0.58, 0.58, 0.90]
seg_display.LookupTable = lut
seg_display.Opacity = 0.88

for obj, color in [(minima, min_color), (saddles, saddle_color), (maxima, max_color)]:
    d = Show(obj, view)
    d.DiffuseColor = color
    d.AmbientColor = color
    d.Ambient = 0.25
    d.Specular = 0.35
    d.SpecularPower = 25.0

view.CameraParallelProjection = 1
view.CameraPosition = [124.0, 60.0, 400.0]
view.CameraFocalPoint = [124.0, 60.0, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 135.0

Render(view)
SaveScreenshot('/workspace/outputs/fracture_morse_smale.png', view, ImageResolution=[1600, 900])
SaveState('/workspace/outputs/fracture_morse_smale.pvsm')
print('Saved /workspace/outputs/fracture_morse_smale.png and .pvsm')
