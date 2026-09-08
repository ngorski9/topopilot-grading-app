from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import argparse
from pathlib import Path


def pipeline(load_path):
    tensor = LoadSymmetricTensorField(
        load_path=load_path,
        A_array_name="A",
        B_array_name="B",
        D_array_name="D",
    )
    tensor_view = EmbedSymmetricTensorField(parent_node_id=tensor.id)
    degenerates = ComputeSymmetricTensorDegeneratePoints(parent_node_id=tensor.id)
    degenerate_view = EmbedSymmetricTensorDegeneratePoints(
        parent_node_id=degenerates.id,
        radius=1.0,
        # The ordering is trisectors, then wedges.
        color_scheme=["#ff8cc6", "#ffffff"],
    )
    return [tensor_view, degenerate_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [53.5, 32.5, 0.0]
    view.CameraPosition = [53.5, 32.5, 100.0]
    view.CameraFocalPoint = [53.5, 32.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 72.0
    view.InteractionMode = "2D"
    return view


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-q", default=".")
    args = parser.parse_args()
    config.set_random_seed(0)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        node.visualize(1, render_view)
    SaveScreenshot(args.o, render_view)
