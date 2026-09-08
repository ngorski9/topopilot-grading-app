from paraview.simple import *
import glob

files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda p: int(p.split('Ionization')[-1].split('.vti')[0]))

# Keep all 21 frames in the source.  The visualization is shown for t=1.
ionization = XMLImageDataReader(registrationName='Ionization time series', FileName=files)
ionization.PointArrayStatus = ['Scalars_']

# Original scalar field: central z slice, coloured with the Viridis preset.
slice1 = Slice(registrationName='Original scalar field (central slice)', Input=ionization)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [149.5, 61.5, 61.5]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]
view = CreateView('RenderView')
view.ViewSize = [1400, 850]
view.Background = [0.08, 0.08, 0.10]
field_display = Show(slice1, view)
ColorBy(field_display, ('POINTS', 'Scalars_'))
field_lut = GetColorTransferFunction('Scalars_')
field_lut.ApplyPreset('Viridis (matplotlib)', True)
field_display.SetScalarBarVisibility(view, True)
field_display.Representation = 'Surface'
view.CameraParallelProjection = 1
view.CameraPosition = [149.5, 61.5, 600]
view.CameraFocalPoint = [149.5, 61.5, 61.5]
view.CameraViewUp = [0, 1, 0]
Render(view)
SaveScreenshot('/workspace/ionization_original_viridis.png', view, ImageResolution=[1400,850])

# Persistence simplification (absolute value 0.1), followed by the contour tree.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.1)', Input=ionization)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
# TTK's persistence value is normalized to the scalar-field range.
simplified.ThresholdIsAbsolute = 0
tree = TTKContourTree(registrationName='Simplified piecewise-linear contour tree', Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0
UpdatePipeline(proxy=tree)

Hide(slice1, view)
# Port 1 contains the contour-tree arcs, port 0 contains vertices.
arcs = OutputPort(tree, 1)
nodes = OutputPort(tree, 0)
arc_display = Show(arcs, view)
arc_display.Representation = 'Surface'
arc_display.DiffuseColor = [0.72, 0.72, 0.72]
arc_display.LineWidth = 1.0

def critical_nodes(value, name, color):
    sel = Threshold(registrationName=name, Input=nodes)
    sel.Scalars = ['POINTS', 'CriticalType']
    sel.ThresholdMethod = 'Between'
    sel.LowerThreshold = value
    sel.UpperThreshold = value
    glyph = Glyph(registrationName=name + ' (radius 2)', Input=sel, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphMode = 'All Points'
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    disp = Show(glyph, view)
    disp.DiffuseColor = color
    disp.AmbientColor = color
    disp.Ambient = 0.35
    return glyph

# TTK CriticalType codes: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
mins = critical_nodes(0, 'Minima', [0.12, 0.35, 1.0])
saddle1 = critical_nodes(1, '1-saddles', [1.0, 1.0, 1.0])
saddle2 = critical_nodes(2, '2-saddles', [1.0, 0.45, 0.05])
maxs = critical_nodes(3, 'Maxima', [1.0, 0.05, 0.05])

view.CameraParallelProjection = 0
view.CameraPosition = [520, -510, 390]
view.CameraFocalPoint = [149.5, 61.5, 61.5]
view.CameraViewUp = [0, 0, 1]
Render(view)
SaveScreenshot('/workspace/ionization_simplified_contour_tree.png', view, ImageResolution=[1400,850])
SaveState('/workspace/ionization_contour_tree.pvsm')
print('Rendered original scalar field and simplified contour tree.')
