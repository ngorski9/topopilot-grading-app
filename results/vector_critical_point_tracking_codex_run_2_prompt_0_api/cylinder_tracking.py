from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    # The loader recognizes the cylinder directory as a time-varying VTI series.
    vector_field = LoadVectorField(
        load_path=load_path,
        u_array_name="u",
        v_array_name="v",
    )
    vector_view = EmbedVectorField(parent_node_id=vector_field.id)

    critical_points = ComputeVectorCriticalPoints(parent_node_id=vector_field.id)
    tracked_points = TrackVectorCriticalPoints(
        parent_node_id=critical_points.id,
        time_steps=3,
        backend="partial",
    )
    tracked_view = EmbedTrackedVectorFieldCriticalPoints(
        parent_node_id=tracked_points.id,
        radius=5.0,
    )

    return [vector_view, tracked_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [900, 900]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [74.5, 224.5, 0.0]
    view.CameraPosition = [74.5, 224.5, 100.0]
    view.CameraFocalPoint = [74.5, 224.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 240.0
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
    if args.t > time_steps:
        print("error: time step is too high")
        exit(1)
    if args.o == "":
        print("error: an output file is required")
        exit(1)
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=3)
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
    for node in nodes_to_visualize:
        node.visualize(args.t, render_view)
    SaveScreenshot(args.o, render_view)
