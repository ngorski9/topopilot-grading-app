from paraview.simple import *

# Reproducible ParaView/TTK pipeline for fracture.vti.
LoadDistributedPlugin('TopologyToolKit', ns=globals())

reader = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])

simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.05)', Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 1

msc = TTKMorseSmaleComplex(registrationName='Morse-Smale complex', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.ThresholdIsAbsolute = 1
msc.SaddleConnectorsPersistenceThreshold = 0.05

# Output port 3 is the piecewise-linear Morse-Smale segmentation; port 0 holds
# the critical points, classified by CellDimension (0=min, 1=saddle, 2=max).
segmentation = OutputPort(msc, 3)
critical = OutputPort(msc, 0)

mins = Threshold(registrationName='Minimums', Input=critical)
mins.Scalars = ['POINTS', 'CellDimension']
mins.ThresholdMethod = 'Between'
mins.LowerThreshold = 0
mins.UpperThreshold = 0

saddles = Threshold(registrationName='Saddles', Input=critical)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.ThresholdMethod = 'Between'
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1

maxs = Threshold(registrationName='Maximums', Input=critical)
maxs.Scalars = ['POINTS', 'CellDimension']
maxs.ThresholdMethod = 'Between'
maxs.LowerThreshold = 2
maxs.UpperThreshold = 2

sphere = Sphere(registrationName='Critical point marker (radius 2)')
sphere.Radius = 2.0
sphere.ThetaResolution = 20
sphere.PhiResolution = 20

min_glyphs = Glyph(registrationName='Minimum markers (r=2)', Input=mins, GlyphType=sphere)
min_glyphs.OrientationArray = ['POINTS', 'No orientation array']
min_glyphs.ScaleArray = ['POINTS', 'No scale array']
min_glyphs.ScaleFactor = 1.0

saddle_glyphs = Glyph(registrationName='Saddle markers (r=2)', Input=saddles, GlyphType=sphere)
saddle_glyphs.OrientationArray = ['POINTS', 'No orientation array']
saddle_glyphs.ScaleArray = ['POINTS', 'No scale array']
saddle_glyphs.ScaleFactor = 1.0

max_glyphs = Glyph(registrationName='Maximum markers (r=2)', Input=maxs, GlyphType=sphere)
max_glyphs.OrientationArray = ['POINTS', 'No orientation array']
max_glyphs.ScaleArray = ['POINTS', 'No scale array']
max_glyphs.ScaleFactor = 1.0

view = CreateView('RenderView')
view.ViewSize = [1800, 900]
view.Background = [0.10, 0.12, 0.16]
view.BackgroundColorMode = 'Single Color'
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.InterpolateScalarsBeforeMapping = 0
seg_display.SetScalarBarVisibility(view, False)
seg_lut = GetColorTransferFunction('MorseSmaleManifold')
seg_lut.ApplyPreset('Rainbow Desaturated', True)
seg_lut.NumberOfTableValues = 256

min_display = Show(min_glyphs, view)
min_display.DiffuseColor = [0.05, 0.30, 1.0]
min_display.AmbientColor = [0.05, 0.30, 1.0]
min_display.Ambient = 0.25

saddle_display = Show(saddle_glyphs, view)
saddle_display.DiffuseColor = [1.0, 1.0, 1.0]
saddle_display.AmbientColor = [1.0, 1.0, 1.0]
saddle_display.Ambient = 0.2

max_display = Show(max_glyphs, view)
max_display.DiffuseColor = [1.0, 0.05, 0.05]
max_display.AmbientColor = [1.0, 0.05, 0.05]
max_display.Ambient = 0.25

view.ResetCamera()
view.CameraParallelScale = 70
view.CameraPosition = [124, 60, 400]
view.CameraFocalPoint = [124, 60, 0]
view.CameraViewUp = [0, 1, 0]

UpdatePipeline(proxy=msc)
SaveData('/workspace/fracture_simplified_0.05.vti', proxy=simplified)
SaveData('/workspace/fracture_morse_smale_segmentation_0.05.vti', proxy=segmentation)
SaveData('/workspace/fracture_critical_points_0.05.vtp', proxy=critical)
SaveScreenshot('/workspace/fracture_morse_smale_0.05.png', view, ImageResolution=[1800, 900])
SaveState('/workspace/fracture_morse_smale_0.05.pvsm')
