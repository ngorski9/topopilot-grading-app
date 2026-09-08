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
    degenerate = ComputeAsymmetricTensorDegeneratePoints(parent_node_id=tensor.id)
    degenerate_view = EmbedAsymmetricTensorDegeneratePoints(
        parent_node_id=degenerate.id,
        radius=1.0,
        # TopoPilot orders these as trisectors, then wedges.
        color_scheme=["#ff8cc6", "#ffffff"],
    )
    return [partition, degenerate_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    # Ocean spans 100 by 100 unit squares; 1000 pixels gives 10 pixels per square.
    view.ViewSize = [1000, 1000]
    view.CenterOfRotation = [50.0, 50.0, 0.0]
    view.CameraPosition = [50.0, 50.0, 100.0]
    view.CameraFocalPoint = [50.0, 50.0, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 50.0
    view.InteractionMode = "2D"
    return view


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-q", default=".topopilot_ocean")
    args = parser.parse_args()

    config.change_session_folder(args.q)
    Path(args.q).mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        node.visualize(1, render_view)
    SaveScreenshot(args.o, render_view)
