from paraview.simple import *

# Reproducible ParaView/TTK pipeline for fracture.vti.

reader = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']

# Persistence simplification of the input scalar field.  The threshold is
# relative (the requested 0.05 persistence fraction), matching TTK's default.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.05)', Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 0
simplified.DebugLevel = 0

# The Morse--Smale complex is computed on the simplified piecewise-linear field.
msc = TTKMorseSmaleComplex(registrationName='Morse-Smale complex', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.DebugLevel = 0

# Output port 3 is the piecewise-linear Morse--Smale segmentation; port 0
# contains critical points, whose CellDimension is 0=min, 1=saddle, 2=max.
segmentation = ExtractSurface(registrationName='Piecewise-linear segmentation', Input=OutputPort(msc, 3))
critical_points = OutputPort(msc, 0)

def select_critical_type(name, cell_dimension):
    selected = Threshold(registrationName=name, Input=critical_points)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.LowerThreshold = cell_dimension
    selected.UpperThreshold = cell_dimension
    selected.ThresholdMethod = 'Between'
    spheres = TTKIcospheresFromPoints(registrationName=name + ' (radius 2)', Input=selected)
    spheres.Radius = 2.0
    spheres.Subdivisions = 3
    spheres.DebugLevel = 0
    return spheres

minima = select_critical_type('Minima', 0)
saddles = select_critical_type('Saddles', 1)
maxima = select_critical_type('Maxima', 2)

view = CreateView('RenderView')
view.ViewSize = [1400, 720]
view.Background = [0.08, 0.08, 0.10]
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1

seg_display = Show(segmentation, view, 'GeometryRepresentation')
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.SetScalarBarVisibility(view, True)
seg_display.Opacity = 0.92
seg_display.Interpolation = 'Flat'
seg_display.Ambient = 1.0
seg_display.Diffuse = 0.0
lut = GetColorTransferFunction('MorseSmaleManifold')
lut.ApplyPreset('Rainbow Desaturated', True)
lut.Discretize = 1
lut.NumberOfTableValues = 16
seg_display.RescaleTransferFunctionToDataRange(True, False)
scalar_bar = GetScalarBar(lut, view)
scalar_bar.Title = 'Morse-Smale\nregion'
scalar_bar.ComponentTitle = ''
scalar_bar.Visibility = 1

for source, color in ((minima, [0.10, 0.35, 1.0]), (saddles, [1.0, 1.0, 1.0]), (maxima, [1.0, 0.05, 0.05])):
    display = Show(source, view, 'GeometryRepresentation')
    display.SetScalarColoring(None, 0)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.30
    display.Diffuse = 0.70
    display.Specular = 0.20
    display.SpecularPower = 20.0

# Use a top-down view of this 2D field and write an immediately usable artifact.
view.ResetCamera()
view.CameraPosition = [124.0, 60.0, 700.0]
view.CameraFocalPoint = [124.0, 60.0, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 145.0
Render(view)
SaveScreenshot('/workspace/output/fracture_morse_smale.png', view, ImageResolution=[1400, 720])
SaveState('/workspace/output/fracture_morse_smale.pvsm')

UpdatePipeline(proxy=msc)
print('Critical points after simplification: %d' % critical_points.GetDataInformation().GetNumberOfPoints())
print('Outputs written to output/fracture_morse_smale.png and .pvsm')
