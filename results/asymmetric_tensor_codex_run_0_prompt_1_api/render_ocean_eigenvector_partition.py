from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import argparse
from pathlib import Path


def pipeline(load_path):
    tensor = LoadAsymmetricTensorField(
        load_path=load_path,
        A_array_name="A",
        B_array_name="B",
        C_array_name="C",
        D_array_name="D",
    )
    partition = EmbedAsymmetricTensorFieldEigenvectorPartition(
        parent_node_id=tensor.id,
        resolution=10,
    )
    degenerate_points = ComputeAsymmetricTensorDegeneratePoints(
        parent_node_id=tensor.id,
    )
    points = EmbedAsymmetricTensorDegeneratePoints(
        parent_node_id=degenerate_points.id,
        radius=1.0,
        # TopoPilot orders these colors as trisectors, then wedges.
        color_scheme=["#ff8cc6", "#ffffff"],
    )
    return [partition, points], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1000, 1000]
    view.OrientationAxesVisibility = 0
    view.CenterOfRotation = [50.0, 50.0, 0.0]
    view.CameraPosition = [50.0, 50.0, 100.0]
    view.CameraFocalPoint = [50.0, 50.0, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 52.0
    view.InteractionMode = "2D"
    return view


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-q", default=".")
    args = parser.parse_args()

    config.change_session_folder(args.q)
    Path(args.q).mkdir(parents=True, exist_ok=True)
    Path(args.q, "cache").mkdir(exist_ok=True)
    nodes_to_visualize, _, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        node.visualize(1, render_view)
    SaveScreenshot(args.o, render_view)
