from paraview.simple import *
import glob
import os
import re

OUT = '/workspace/Ionization_renderings'
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda p: int(re.search(r'(\d+)', p).group(1)))

# Import every time step as one temporal VTI source.
ionization = XMLImageDataReader(FileName=files)
ionization.PointArrayStatus = ['Scalars_']
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
time_value = scene.TimeKeeper.TimestepValues[0]

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [0.10, 0.10, 0.12]
view.UseColorPaletteForBackground = 0

# Original scalar field, rendered as a viridis-colored volume.
original = Show(ionization, view)
original.Representation = 'Volume'
ColorBy(original, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Viridis (matplotlib)', True)
original.RescaleTransferFunctionToDataRange(True, False)
original.ScalarOpacityUnitDistance = 1.0
original.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
view.ViewTime = time_value
Render(view)
ResetCamera(view)
SaveScreenshot(os.path.join(OUT, 'ionization_original_viridis.png'), view, ImageResolution=[1400, 1000])

# Persistence simplification (absolute persistence 0.1), followed by its PL contour tree.
Hide(ionization, view)
simplified = TTKTopologicalSimplificationByPersistence(Input=ionization)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 1
simplified.UseAllCores = 1

tree = TTKContourTree(Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 1
tree.UseAllCores = 1

# Output port 1 is the tree arcs.  It remains a true piecewise-linear tree.
arcs = OutputPort(tree, 1)
arcs_display = Show(arcs, view)
arcs_display.Representation = 'Surface'
arcs_display.DiffuseColor = [0.72, 0.72, 0.72]
arcs_display.LineWidth = 1.0

# Output port 0 contains critical nodes.  CriticalType: min=0, 1-saddle=1,
# 2-saddle=2, max=3.  Render every category as radius-2 sphere glyphs.
nodes = OutputPort(tree, 0)
def critical_nodes(value, name, color):
    selected = Threshold(Input=nodes)
    selected.Scalars = ['POINTS', 'CriticalType']
    selected.ThresholdMethod = 'Between'
    selected.LowerThreshold = value
    selected.UpperThreshold = value
    glyph = Glyph(Input=selected, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 16
    glyph.GlyphType.PhiResolution = 16
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.GlyphMode = 'All Points'
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.25
    display.Specular = 0.15
    return glyph

minima = critical_nodes(0, 'Minima', [0.12, 0.35, 1.0])
saddle1 = critical_nodes(1, 'Saddle1', [1.0, 1.0, 1.0])
saddle2 = critical_nodes(2, 'Saddle2', [1.0, 0.50, 0.05])
maxima = critical_nodes(3, 'Maxima', [0.95, 0.10, 0.08])

view.ViewTime = time_value
Render(view)
ResetCamera(view)
SaveScreenshot(os.path.join(OUT, 'ionization_simplified_contour_tree.png'), view, ImageResolution=[1400, 1000])

# Persist the full, editable ParaView pipeline and display configuration.
SaveState(os.path.join(OUT, 'ionization_contour_tree.pvsm'))
print('Saved:', OUT)
print('Frames imported:', len(files), 'time steps:', len(scene.TimeKeeper.TimestepValues))
print('Persistence threshold:', simplified.PersistenceThreshold, '(absolute)')
