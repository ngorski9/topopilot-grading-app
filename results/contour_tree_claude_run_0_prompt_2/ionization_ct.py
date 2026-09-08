import re, glob, os
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

plugin = '/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so'
LoadPlugin(plugin, remote=False, ns=globals())

files = glob.glob('/workspace/Ionization/Ionization*.vti')
files.sort(key=lambda f: int(re.search(r'(\d+)\.vti$', f).group(1)))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.TimeArray = 'None'
reader.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

# --- 1. Raw scalar field, viridis colormap ---
rawDisplay = Show(reader, view)
rawDisplay.SetRepresentationType('Volume')
ColorBy(rawDisplay, ('POINTS', 'Scalars_'))
rawDisplay.SetScalarBarVisibility(view, True)
rawLUT = GetColorTransferFunction('Scalars_')
rawLUT.ApplyPreset('Viridis (matplotlib)', True)
rawDisplay.RescaleTransferFunctionToDataRange(False, True)
view.ResetCamera()
Render(view)
SaveScreenshot('/workspace/out/01_raw_scalar_field_viridis.png', view)

Hide(reader, view)

# --- 2. Persistence simplification (threshold 0.1) ---
simp = TTKTopologicalSimplificationByPersistence(Input=reader)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.1
simp.ThresholdIsAbsolute = 1
simp.UpdatePipeline()

# --- 3. Contour tree (piecewise-linear) on the simplified field ---
ct = TTKContourTree(Input=simp)
ct.ScalarField = ['POINTS', 'Scalars_']
ct.UpdatePipeline()

# --- 4. Visualize simplified contour tree: edges as tubes (radius 1), nodes as spheres (radius 2) ---
surf = ExtractSurface(Input=ct)
tube = Tube(Input=surf)
tube.Radius = 1.0
tube.VaryRadius = 'Off'
tubeDisplay = Show(tube, view)
tubeDisplay.SetRepresentationType('Surface')
tubeDisplay.ColorArrayName = [None, '']
tubeDisplay.AmbientColor = [0.7, 0.7, 0.7]
tubeDisplay.DiffuseColor = [0.7, 0.7, 0.7]

glyph = Glyph(Input=surf, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'
glyph.GlyphTransform = 'Transform2'
glyphDisplay = Show(glyph, view)
ColorBy(glyphDisplay, ('POINTS', 'CriticalType'))
glyphDisplay.SetScalarBarVisibility(view, True)

ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1
# CriticalType: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
ctLUT.Annotations = ['0', 'minimum', '1', '1-saddle', '2', '2-saddle', '3', 'maximum']
ctLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]

view.ResetCamera()
Render(view)
SaveScreenshot('/workspace/out/02_simplified_contour_tree.png', view)

print('DONE')
