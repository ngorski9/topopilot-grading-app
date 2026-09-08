from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    # Keep the original field for the requested raw-field rendering.
    raw_scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    raw_field_view = EmbedScalarField(
        parent_node_id=raw_scalar.id,
        colormap="viridis",
    )

    # Build the requested piecewise-linear contour tree from the simplified field.
    simplified_scalar = PersistenceSimplification(
        parent_node_id=raw_scalar.id,
        epsilon=0.1,
    )
    contour_tree = ComputeContourTree(parent_node_id=simplified_scalar.id)
    contour_tree_view = EmbedContourTree(
        parent_node_id=contour_tree.id,
        tube_radius=1.0,
        ball_radius=2.0,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [raw_field_view, contour_tree_view], {
        raw_field_view.id: ([0.84, 0.15], 0.70, "Ionization"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.CenterOfRotation = [149.5, 61.5, 61.5]
    view.CameraPosition = [520.0, 430.0, 390.0]
    view.CameraFocalPoint = [149.5, 61.5, 61.5]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = False
    view.CameraViewAngle = 30.0
    view.InteractionMode = "3D"
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
                print(f"error: missing file {dirname}/{basename}/{basename}{i}.vti")
                exit(1)
    if args.t > time_steps:
        print("error: time step is too high")
        exit(1)
    if args.x and args.t == -1 and time_steps > 1:
        print("error: cannot run interactive mode on multiple time steps")
        exit(1)
    if args.o != "" and args.x:
        print("error: you cannot specify an output file and -interactive")
        exit(1)
    if args.o == "" and not args.x:
        print("error: you must either specify an output file or run in interactive mode.")
        exit(1)
    if args.t == -1 and time_steps > 1 and args.o[-4:] != ".mp4":
        print("error: video output filename must end in .mp4")
        exit(1)
    if args.t != -1 and not args.x and args.o[-4:] != ".png":
        print("error: image output filename must end in .png")
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

    time_steps = validate_arguments(args)
    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)

    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_steps = range(args.t, args.t + 1) if args.t != -1 else range(1, time_steps + 1)

    for t in render_steps:
        render_view = create_view()
        for node in nodes_to_visualize:
            props = node.visualize(t, render_view)
            if node.id in color_bar_info:
                position, length, title = color_bar_info[node.id]
                bar = GetScalarBar(props.LookupTable, render_view)
                bar.WindowLocation = "Any Location"
                bar.Position = position
                bar.ScalarBarLength = length
                bar.Title = title
                bar.ComponentTitle = ""
                props.SetScalarBarVisibility(render_view, True)
        if args.t != -1:
            SaveScreenshot(args.o, render_view)
        else:
            SaveScreenshot(f"{args.q}/frame{t:08d}.png", render_view)
            Disconnect()
            tree.clear_evaluation_cache()
            Connect()

    if args.t == -1:
        os.system(f"ffmpeg -y -framerate 4 -i {args.q}/frame%08d.png -c:v libx264 -pix_fmt yuv420p {args.o}")
        for t in render_steps:
            os.remove(f"{args.q}/frame{t:08d}.png")
