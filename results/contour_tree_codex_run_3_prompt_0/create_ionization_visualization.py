from paraview.simple import *

# Time-varying VTI collection (one VTI file per time step).
files = [f'/workspace/Ionization/Ionization{i}.vti' for i in range(1, 22)]
source = XMLImageDataReader(registrationName='Ionization (time varying)', FileName=files)

# Persistence simplification at the requested normalized threshold, followed by
# a contour tree of the simplified scalar field.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (threshold 0.1)', Input=source)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 0

tree = TTKContourTree(registrationName='Simplified piecewise-linear contour tree', Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0

# ContourTree ports: 0 = nodes, 1 = arcs, 2 = segmentation.
nodes = OutputPort(tree, 0)
arcs = OutputPort(tree, 1)

# Render each contour-tree arc as a tube of radius 1.
arc_lines = ExtractSurface(registrationName='Contour-tree arc lines', Input=arcs)
tubes = Tube(registrationName='Contour-tree edges (radius 1)', Input=arc_lines)
tubes.Radius = 1.0
tubes.NumberofSides = 12

# Critical types emitted by TTK: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
critical = [
    ('Minima (blue)', 0, [0.0, 0.25, 1.0]),
    ('1-saddles (white)', 1, [1.0, 1.0, 1.0]),
    ('2-saddles (orange)', 2, [1.0, 0.45, 0.0]),
    ('Maxima (red)', 3, [1.0, 0.0, 0.0]),
]
glyphs = []
for label, value, color in critical:
    selected = Threshold(registrationName=label, Input=nodes)
    selected.Scalars = ['POINTS', 'CriticalType']
    selected.LowerThreshold = value
    selected.UpperThreshold = value
    selected.ThresholdMethod = 'Between'
    spheres = Glyph(registrationName=label + ' vertices (radius 2)', Input=selected,
                    GlyphType='Sphere')
    spheres.GlyphType.Radius = 2.0
    spheres.GlyphType.ThetaResolution = 16
    spheres.GlyphType.PhiResolution = 16
    spheres.ScaleArray = ['POINTS', 'No scale array']
    spheres.ScaleFactor = 1.0
    glyphs.append((spheres, color))

# Original field: central planar slice makes the scalar field legible, colored
# with the requested viridis preset.
field_slice = Slice(registrationName='Original scalar field', Input=source)
field_slice.SliceType = 'Plane'
field_slice.SliceType.Origin = [149.5, 61.5, 61.5]
field_slice.SliceType.Normal = [0.0, 0.0, 1.0]

layout = CreateLayout('Ionization scalar field and simplified contour tree')
field_view = CreateView('RenderView')
tree_view = CreateView('RenderView')
layout.SplitHorizontal(0, 0.5)
layout.AssignView(1, field_view)
layout.AssignView(2, tree_view)

field_view.ViewSize = [900, 800]
field_view.Background = [0.08, 0.08, 0.10]
field_view.OrientationAxesVisibility = 0
field_display = Show(field_slice, field_view, 'GeometryRepresentation')
ColorBy(field_display, ('POINTS', 'Scalars_'))
field_lut = GetColorTransferFunction('Scalars_')
field_lut.ApplyPreset('Viridis (matplotlib)', True)
field_display.SetScalarBarVisibility(field_view, True)
field_display.Representation = 'Surface'
field_display.RescaleTransferFunctionToDataRange(True, False)

tree_view.ViewSize = [900, 800]
tree_view.Background = [0.08, 0.08, 0.10]
tree_view.OrientationAxesVisibility = 0
edge_display = Show(tubes, tree_view, 'GeometryRepresentation')
edge_display.DiffuseColor = [0.72, 0.72, 0.72]
edge_display.AmbientColor = [0.72, 0.72, 0.72]
for spheres, color in glyphs:
    display = Show(spheres, tree_view, 'GeometryRepresentation')
    display.DiffuseColor = color
    display.AmbientColor = color

field_view.ResetCamera()
tree_view.ResetCamera()
field_view.CameraPosition = [149.5, 61.5, 520.0]
field_view.CameraFocalPoint = [149.5, 61.5, 61.5]
field_view.CameraViewUp = [0.0, 1.0, 0.0]

# Save an immediately reusable, time-aware ParaView scene and a rendered proof.
SaveState('/workspace/output/ionization_contour_tree.pvsm')
RenderAllViews()
SaveScreenshot('/workspace/output/ionization_contour_tree.png', layout, ImageResolution=[1800, 800])
SaveScreenshot('/workspace/output/original_scalar_field_viridis.png', field_view, ImageResolution=[900, 800])
SaveScreenshot('/workspace/output/simplified_contour_tree.png', tree_view, ImageResolution=[900, 800])
