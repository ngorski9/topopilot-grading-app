from paraview.simple import *
import os
import vtk
from vtk.util import numpy_support as vnp
import numpy as np

paraview.simple._DisableFirstRenderCameraReset()

# ---- Build a full 9-component tensor array (Txx,Txy,Txz,Tyx,Tyy,...) from
# ---- A, B, D and write it out alongside the original field. ----
_reader = vtk.vtkXMLImageDataReader()
_reader.SetFileName("/workspace/brain.vti")
_reader.Update()
_img = _reader.GetOutput()
_A = vnp.vtk_to_numpy(_img.GetPointData().GetArray("A"))
_B = vnp.vtk_to_numpy(_img.GetPointData().GetArray("B"))
_D = vnp.vtk_to_numpy(_img.GetPointData().GetArray("D"))
_T = np.zeros((_A.shape[0], 9))
_T[:, 0] = _A
_T[:, 1] = _B
_T[:, 3] = _B
_T[:, 4] = _D
_tarr = vnp.numpy_to_vtk(_T, deep=True)
_tarr.SetName("Tensor2D")
_img.GetPointData().AddArray(_tarr)
_writer = vtk.vtkXMLImageDataWriter()
_writer.SetFileName("/workspace/brain_with_tensor.vti")
_writer.SetInputData(_img)
_writer.Write()

# ---- Load the tensor field ----
brain = XMLImageDataReader(FileName=["/workspace/brain.vti"])
brain.PointArrayStatus = ["A", "B", "D"]

calc = XMLImageDataReader(FileName=["/workspace/brain_with_tensor.vti"])
calc.PointArrayStatus = ["A", "B", "D", "Tensor2D"]

# ---- Tensor glyph visualization of the field (subsampled for clarity) ----
merged = MergeBlocks(Input=calc)

maskPts = MaskPoints(Input=merged)
maskPts.OnRatio = 10
maskPts.GenerateVertices = 1

glyph = TensorGlyph(Input=maskPts)
glyph.Tensors = ["POINTS", "Tensor2D"]
glyph.ScaleFactor = 1200.0
glyphDisplay = Show(glyph)
glyphDisplay.Representation = "Surface"
glyphDisplay.ColorArrayName = [None, ""]
glyphDisplay.AmbientColor = [0.55, 0.65, 0.85]
glyphDisplay.DiffuseColor = [0.55, 0.65, 0.85]
glyphDisplay.SetScalarBarVisibility(GetActiveView(), False)

# base slice of the domain for context
outline = Outline(Input=brain)
Show(outline)

# ---- Degenerate points ----
degpts = XMLPolyDataReader(FileName=["/workspace/degenerate_points.vtp"])

trisectors = Threshold(Input=degpts)
trisectors.Scalars = ["POINTS", "DegenerateType"]
trisectors.LowerThreshold = 0
trisectors.UpperThreshold = 0
trisectors.ThresholdMethod = "Between"

wedges = Threshold(Input=degpts)
wedges.Scalars = ["POINTS", "DegenerateType"]
wedges.LowerThreshold = 1
wedges.UpperThreshold = 1
wedges.ThresholdMethod = "Between"

triGlyph = Glyph(Input=trisectors, GlyphType="Sphere")
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = "All Points"
triDisplay = Show(triGlyph)
triDisplay.DiffuseColor = [1.0, 0.4117647058823529, 0.7058823529411765]  # pink

wedgeGlyph = Glyph(Input=wedges, GlyphType="Sphere")
wedgeGlyph.GlyphType.Radius = 1.0
wedgeGlyph.ScaleFactor = 1.0
wedgeGlyph.GlyphMode = "All Points"
wedgeDisplay = Show(wedgeGlyph)
wedgeDisplay.DiffuseColor = [1.0, 1.0, 1.0]  # white
wedgeDisplay.AmbientColor = [1.0, 1.0, 1.0]

view = GetActiveView()
view.Background = [0.1, 0.1, 0.12]
view.OrientationAxesVisibility = 0

ResetCamera()
cam = GetActiveCamera()
cam.Elevation(0)

Render()
SaveScreenshot("/workspace/brain_degenerate_points.png", view, ImageResolution=[1600, 1000])
print("Saved /workspace/brain_degenerate_points.png")

SaveState("/workspace/brain_degenerate_points.pvsm")
print("Saved /workspace/brain_degenerate_points.pvsm")
