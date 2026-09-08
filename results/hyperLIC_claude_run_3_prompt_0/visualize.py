from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# Main dataset (with augmented full 3x3 "Tensor" array for coloring/eigen-glyphing)
field = XMLImageDataReader(FileName=["brain_with_tensor.vti"])
field.PointArrayStatus = ["A", "B", "D", "Tensor"]

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 900]
view.OrientationAxesVisibility = 0
view.Background = [0.12, 0.12, 0.14]

# Show the field itself as a surface colored by the trace of the tensor (A+D)
calc = Calculator(Input=field)
calc.ResultArrayName = "Trace"
calc.Function = "A+D"

fieldDisplay = Show(calc, view)
ColorBy(fieldDisplay, ("POINTS", "Trace"))
fieldDisplay.SetRepresentationType("Surface")
fieldDisplay.RescaleTransferFunctionToDataRange(True)
traceLUT = GetColorTransferFunction("Trace")
traceLUT.ApplyPreset("Cool to Warm", True)
fieldDisplay.SetScalarBarVisibility(view, True)

# Degenerate points: trisectors (pink) and wedges (white)
trisectors = XMLPolyDataReader(FileName=["trisectors.vtp"])
wedges = XMLPolyDataReader(FileName=["wedges.vtp"])

triGlyph = Glyph(Input=trisectors, GlyphType="Sphere")
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = "All Points"
triDisplay = Show(triGlyph, view)
triDisplay.DiffuseColor = [1.0, 0.4118, 0.7059]  # pink
triDisplay.AmbientColor = [1.0, 0.4118, 0.7059]

wedgeGlyph = Glyph(Input=wedges, GlyphType="Sphere")
wedgeGlyph.GlyphType.Radius = 1.0
wedgeGlyph.ScaleFactor = 1.0
wedgeGlyph.GlyphMode = "All Points"
wedgeDisplay = Show(wedgeGlyph, view)
wedgeDisplay.DiffuseColor = [1.0, 1.0, 1.0]  # white
wedgeDisplay.AmbientColor = [1.0, 1.0, 1.0]

ResetCamera(view)
view.CameraPosition = [53.5, 32.5, 260]
view.CameraFocalPoint = [53.5, 32.5, 0]
view.CameraViewUp = [0, 1, 0]
ResetCamera(view)

Render(view)
SaveScreenshot("brain_degenerate_points.png", view, ImageResolution=[1400, 900])
servermanager.SaveState("brain_degenerate_points.pvsm")
print("saved screenshot and state")
