from paraview.simple import *
import glob, os, re

base = '/workspace/Ionization'
files = sorted(glob.glob(base + '/Ionization*.vti'),
               key=lambda p: int(re.search(r'(\d+)\.vti$', p).group(1)))

# Load the complete file series (Ionization1 ... Ionization21).
ionization = XMLImageDataReader(FileName=files)
ionization.PointArrayStatus = ['Scalars_']
ionization.SMProxy.SetAnnotation('name', 'Ionization time series (21 timesteps)')

# Persistence simplification of the scalar field, followed by its contour tree.
simplified = TTKTopologicalSimplificationByPersistence(Input=ionization)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 0
simplified.SMProxy.SetAnnotation('name', 'Persistence simplification (threshold = 0.1)')

tree = TTKContourTree(Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0
tree.SMProxy.SetAnnotation('name', 'Simplified piecewise-linear contour tree')

# Output port 1 contains the tree arcs and output port 0 the critical vertices.
arcs = OutputPort(tree, 1)
nodes = OutputPort(tree, 0)
# TTK emits unstructured grids; convert them to polydata before tubular/glyph geometry.
arc_surface = ExtractSurface(Input=arcs)
node_surface = ExtractSurface(Input=nodes)
edges = Tube(Input=arc_surface)
edges.Radius = 1.0
edges.NumberofSides = 12
edges.Capping = 1
edges.SMProxy.SetAnnotation('name', 'Contour-tree edges (radius 1)')

def critical_points(value, name):
    f = Threshold(Input=node_surface)
    f.Scalars = ['POINTS', 'CriticalType']
    f.LowerThreshold = value
    f.UpperThreshold = value
    f.ThresholdMethod = 'Between'
    f.SMProxy.SetAnnotation('name', name + ' (radius 2)')
    g = Glyph(Input=f, GlyphType='Sphere')
    g.GlyphType.Radius = 2.0
    g.GlyphType.ThetaResolution = 16
    g.GlyphType.PhiResolution = 12
    g.ScaleArray = ['POINTS', 'No scale array']
    g.ScaleFactor = 1.0
    return g

# TTK critical-type convention: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
minima = critical_points(0, 'Minima')
saddle1 = critical_points(1, '1-saddles')
saddle2 = critical_points(2, '2-saddles')
maxima = critical_points(3, 'Maxima')

# Original field: an orthogonal central slice coloured with viridis.
field_slice = Slice(Input=ionization)
field_slice.SliceType = 'Plane'
field_slice.SliceType.Origin = [149.5, 61.5, 61.5]
field_slice.SliceType.Normal = [0.0, 0.0, 1.0]
field_slice.SMProxy.SetAnnotation('name', 'Original Scalars_ field (viridis)')

layout = CreateLayout('Ionization scalar field and simplified contour tree')
left = CreateView('RenderView')
right = CreateView('RenderView')
layout.AssignView(0, left)
layout.SplitHorizontal(0, 0.5)
layout.AssignView(2, right)

left.ViewSize = [900, 700]
left.Background = [0.08, 0.08, 0.10]
field_display = Show(field_slice, left)
ColorBy(field_display, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Viridis (matplotlib)', True)
field_display.SetScalarBarVisibility(left, True)
bar = GetScalarBar(lut, left)
bar.Title = 'Scalars_'
bar.ComponentTitle = ''
left.ResetCamera()
left.CameraPosition = [149.5, 61.5, 620]
left.CameraFocalPoint = [149.5, 61.5, 61.5]
left.CameraViewUp = [0, 1, 0]

right.ViewSize = [900, 700]
right.Background = [0.08, 0.08, 0.10]
ed = Show(edges, right)
ed.DiffuseColor = [0.72, 0.72, 0.72]
ed.Ambient = 0.25
for src, color in [(maxima, [1.0, 0.0, 0.0]),
                   (saddle2, [1.0, 0.5, 0.0]),
                   (saddle1, [1.0, 1.0, 1.0]),
                   (minima, [0.0, 0.3, 1.0])]:
    d = Show(src, right)
    d.DiffuseColor = color
    d.Ambient = 0.35
right.ResetCamera()

# Titles make the two requested outputs unambiguous in the saved image/state.
title1 = Text(registrationName='Original scalar field — viridis')
title1.Text = 'Original scalar field — viridis'
t1 = Show(title1, left); t1.FontSize = 20; t1.Color = [1, 1, 1]; t1.Position = [0.02, 0.94]
title2 = Text(registrationName='Simplified contour tree — persistence 0.1')
title2.Text = 'Simplified contour tree — persistence threshold 0.1'
t2 = Show(title2, right); t2.FontSize = 20; t2.Color = [1, 1, 1]; t2.Position = [0.02, 0.94]

RenderAllViews()
SaveScreenshot('/workspace/ionization_contour_tree.png', layout, SaveAllViews=1, ImageResolution=[1800, 700])
SaveState('/workspace/ionization_contour_tree.pvsm')
print('Saved /workspace/ionization_contour_tree.pvsm and /workspace/ionization_contour_tree.png')
