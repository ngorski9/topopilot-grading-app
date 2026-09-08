from paraview.simple import *

field = XMLImageDataReader(registrationName='Brain tensor field', FileName=['brain_tensor.vti'])
field_display = Show(field)
field_display.Representation = 'Surface'
ColorBy(field_display, ('POINTS', 'A'))
field_display.LookupTable = GetColorTransferFunction('A')
field_display.Opacity = 0.58

# A sparse, symmetric two-eigenvector depiction of the tensor field.
sample = MaskPoints(registrationName='Tensor field sample', Input=field)
sample.MaximumNumberofPoints = 900
sample.RandomSampling = 0
glyphs = TensorGlyph(registrationName='Tensor field glyphs', Input=sample)
glyphs.Tensors = ['POINTS', 'Tensor']
glyphs.Symmetric = 1
glyphs.ThreeGlyphs = 0
glyphs.ScaleFactor = 0.55
glyphs.MaxScaleFactor = 1.0
glyphs_display = Show(glyphs)
glyphs_display.DiffuseColor = [0.16, 0.25, 0.36]
glyphs_display.LineWidth = 1.2
glyphs_display.Translation = [0, 0, 0.15]

def singularity_layer(name, path, color):
    points = XMLPolyDataReader(registrationName=name + ' points', FileName=[path])
    marks = Glyph(registrationName=name, Input=points, GlyphType='Sphere')
    marks.GlyphType.Radius = 1.0
    marks.GlyphType.ThetaResolution = 18
    marks.GlyphType.PhiResolution = 12
    marks.ScaleFactor = 1.0
    marks_display = Show(marks)
    marks_display.DiffuseColor = color
    marks_display.AmbientColor = color
    marks_display.Ambient = 0.35
    marks_display.Specular = 0.25
    marks_display.Translation = [0, 0, 0.35]
    return marks

trisectors = singularity_layer('Trisectors (radius 1)', 'brain_trisectors.vtp', [1.0, 0.35, 0.68])
wedges = singularity_layer('Wedges (radius 1)', 'brain_wedges.vtp', [1.0, 1.0, 1.0])

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 900]
view.Background = [0.045, 0.055, 0.075]
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
view.ResetCamera()
view.CameraParallelScale = 38
Render()
SaveScreenshot('brain_degenerate_points.png', view, ImageResolution=[1400, 900])
SaveState('brain_degenerate_points.pvsm')
