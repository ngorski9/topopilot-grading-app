from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    tensor = LoadAsymmetricTensorField(
        load_path=load_path,
        A_array_name="A",
        B_array_name="B",
        C_array_name="C",
        D_array_name="D",
    )
    degenerate_points = ComputeAsymmetricTensorDegeneratePoints(
        parent_node_id=tensor.id,
    )
    partition_view = EmbedAsymmetricTensorFieldEigenvectorPartition(
        parent_node_id=tensor.id,
        resolution=10,
    )
    points_view = EmbedAsymmetricTensorDegeneratePoints(
        parent_node_id=degenerate_points.id,
        radius=1.0,
        color_scheme=["#ff8cc6", "#ffffff"],  # trisectors, wedges
    )
    return [partition_view, points_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1200, 1200]
    view.OrientationAxesVisibility = 0
    view.CenterOfRotation = [50.0, 50.0, 0.0]
    view.CameraPosition = [50.0, 50.0, 100.0]
    view.CameraFocalPoint = [50.0, 50.0, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 55.0
    view.InteractionMode = "2D"
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            print("error: file specified with -i must either be .vti file or directory")
            exit(1)
        time_steps = len(os.listdir(args.i))
        basename = os.path.basename(args.i)
        dirname = os.path.dirname(args.i)
        for i in range(1, time_steps + 1):
            if not os.path.isfile(f"{dirname}/{basename}/{basename}{i}.vti"):
                print("error: invalid input format for directory")
                exit(1)
    if args.t > time_steps or (args.o == "" and not args.x) or (args.o != "" and args.x):
        print("error: invalid arguments")
        exit(1)
    if args.t == -1 and time_steps == 1:
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-x", action=argparse.BooleanOptionalAction)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-q", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    validate_arguments(args)
    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        node.visualize(args.t, render_view)
    if args.o:
        SaveScreenshot(args.o, render_view)
    else:
        Interact()
