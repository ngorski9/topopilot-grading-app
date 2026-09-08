from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

import glob, os

# ------------------------------------------------------------------
# Load the time-varying Ionization dataset
# ------------------------------------------------------------------
files = sorted(
    glob.glob('/workspace/Ionization/Ionization*.vti'),
    key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))))
)
reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.OrientationAxesVisibility = 0

# ------------------------------------------------------------------
# 1) Visualize the original scalar field with the viridis colormap
# ------------------------------------------------------------------
readerDisplay = Show(reader, renderView1)
readerDisplay.Representation = 'Volume'
ColorBy(readerDisplay, ('POINTS', 'Scalars_'))
readerDisplay.SetScalarBarVisibility(renderView1, True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/Ionization_original_scalar_field.png', renderView1,
               ImageResolution=[1400, 1000])

Hide(reader, renderView1)

# ------------------------------------------------------------------
# 2) Persistence simplification (threshold = 0.1) + contour tree
# ------------------------------------------------------------------
tetra = Tetrahedralize(Input=reader)

simplification = TTKTopologicalSimplificationByPersistence(Input=tetra)
simplification.InputArray = ['POINTS', 'Scalars_']
simplification.PersistenceThreshold = 0.1
simplification.ThresholdIsAbsolute = 0

contourTree = TTKContourTree(Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# Freeze the computed tree into a static source so downstream filters
# (Tube/Glyph) never trigger a re-execution of the expensive TTK pipeline.
frozenTreeData = servermanager.Fetch(contourTree)
frozenTree = TrivialProducer(registrationName='FrozenContourTree')
frozenTree.GetClientSideObject().SetOutput(frozenTreeData)
frozenTree.UpdatePipeline()

# ------------------------------------------------------------------
# 3) Draw the tree edges as tubes (radius = 1)
# ------------------------------------------------------------------
geom = ExtractSurface(Input=frozenTree)

tube = Tube(Input=geom)
tube.Radius = 1.0

tubeDisplay = Show(tube, renderView1)
tubeDisplay.SetRepresentationType('Surface')
tubeDisplay.ColorArrayName = [None, '']
tubeDisplay.AmbientColor = [0.6, 0.6, 0.6]
tubeDisplay.DiffuseColor = [0.6, 0.6, 0.6]

# ------------------------------------------------------------------
# 4) Draw the tree vertices as spheres (radius = 2), colored by
#    critical type: maxima=red, 2-saddles=orange, 1-saddles=white,
#    minima=blue
#    TTK CriticalType convention: 0=minimum, 1=1-saddle,
#    2=2-saddle, 3=maximum
# ------------------------------------------------------------------
glyph = Glyph(Input=frozenTree, GlyphType='Sphere')
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 4.0  # diameter = 2 * radius(2)
glyph.GlyphMode = 'All Points'
glyph.GlyphType.Radius = 2.0

glyphDisplay = Show(glyph, renderView1)
ColorBy(glyphDisplay, ('POINTS', 'CriticalType'))
glyphDisplay.SetScalarBarVisibility(renderView1, True)

criticalTypeLUT = GetColorTransferFunction('CriticalType')
criticalTypeLUT.InterpretValuesAsCategories = 1
criticalTypeLUT.AnnotationsInitialized = 1
criticalTypeLUT.Annotations = ['0', 'minimum', '1', '1-saddle', '2', '2-saddle', '3', 'maximum']
criticalTypeLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # 0: minimum -> blue
    1.0, 1.0, 1.0,   # 1: 1-saddle -> white
    1.0, 0.5, 0.0,   # 2: 2-saddle -> orange
    1.0, 0.0, 0.0,   # 3: maximum -> red
]

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/Ionization_simplified_contour_tree.png', renderView1,
               ImageResolution=[1400, 1000])

print("Done. Saved:")
print("  /workspace/Ionization_original_scalar_field.png")
print("  /workspace/Ionization_simplified_contour_tree.png")
