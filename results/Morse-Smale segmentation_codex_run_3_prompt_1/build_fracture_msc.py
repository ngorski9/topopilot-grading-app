from paraview.simple import *

# Simplify the input scalar field at an absolute persistence of 0.05.
reader = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.05)', Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 1

# Piecewise-linear Morse--Smale complex of the simplified field.
msc = TTKMorseSmaleComplex(registrationName='Morse-Smale complex (simplified)', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1
msc.UpdatePipeline()

# The fourth output is the piecewise-linear segmentation; output zero contains
# the critical points.  Split points by their TTK CellDimension classification.
segmentation = OutputPort(msc, 3)
critical = OutputPort(msc, 0)

def critical_type(name, dimension):
    f = Threshold(registrationName=name, Input=critical)
    f.Scalars = ['POINTS', 'CellDimension']
    f.ThresholdMethod = 'Between'
    f.LowerThreshold = dimension
    f.UpperThreshold = dimension
    g = Glyph(registrationName=name + ' (radius 2)', Input=f, GlyphType='Sphere')
    g.GlyphType.Radius = 2.0
    g.ScaleArray = ['POINTS', 'No scale array']
    g.ScaleFactor = 1.0
    return g

mins = critical_type('Minima', 0)
saddles = critical_type('Saddles', 1)
maxes = critical_type('Maxima', 2)

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [0.08, 0.08, 0.08]

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.SetScalarBarVisibility(view, True)
seg_display.Opacity = 0.82

for source, color in ((mins, [0.0, 0.25, 1.0]), (saddles, [1.0, 1.0, 1.0]), (maxes, [1.0, 0.0, 0.0])):
    display = Show(source, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.RenderPointsAsSpheres = 1

Hide(reader, view)
Hide(simplified, view)
ResetCamera(view)
view.CameraParallelProjection = 1
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view)
SaveState('/workspace/fracture_morse_smale.pvsm')
