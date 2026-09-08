from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0

def pipeline(load_path):
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    simplified = PersistenceSimplification(
        parent_node_id=scalar.id,
        epsilon=0.05,
    )

    segmentation = ComputeMorseSmaleSegmentation(parent_node_id=simplified.id)
    segmentation_view = EmbedMorseSmaleSegmentation(
        parent_node_id=segmentation.id,
    )

    critical_points = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="all critical points (mins+maxes+saddles)",
    )
    critical_points_view = EmbedScalarCriticalPoints(
        parent_node_id=critical_points.id,
        radius=2.0,
        color_scheme=["#0000ff", "#ffffff", "#ffffff", "#ff0000", "#ffffff"],
    )

    return [segmentation_view, critical_points_view], {}, [], []

def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [124.0, 60.0, 0.0]
    view.CameraPosition = [124.0, 60.0, 400.0]
    view.CameraFocalPoint = [124.0, 60.0, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 135.0
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

        for i in range(1,time_steps+1):
            
            if not os.path.isfile(f"{dirname}/{basename}/{basename}{i}.vti"):
                print("error: invalid input format for directory")
                print(f"missing file {dirname}/{basename}/{basename}{i}.vti")
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
        print("error: when outputting to video format your output filename must end in .mp4")
        exit(1)
    
    if args.t != -1 and not args.x and args.o[-4:] != ".png":
        print("error: when outputting to image format your output filename must end in .png")
        exit(1)
                                    
    if args.t == -1 and time_steps == 1:
        args.t = 1

    return time_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User generated scientific visualization pipeline. Generates images or videos based on user input.")

    parser.add_argument("-i", help="The file that should be loaded.", required=True)
    parser.add_argument("-x", help="Should you load into interactive mode? (no images will be generated)", action=argparse.BooleanOptionalAction)
    parser.add_argument("-o", help="The name of the file that we are outputting.", default="")
    parser.add_argument("-t", help="What time step should be loaded. Put -1 to generate a video of all time steps. If the data only has one time step, -1 will just load the one time step.", type=int, default=-1)
    parser.add_argument("-q", help="What folder should be used for temporary files", default=".")
    parser.add_argument("--seed", help="Seed for TopoPilot-owned random colors and identifiers.", type=int, default=DEFAULT_RANDOM_SEED)

    args = parser.parse_args()

    time_steps = validate_arguments(args)

    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)

    sessionDir = Path(f"{config.session_folder()}")
    sessionDir.mkdir(exist_ok = True)
    cacheDir = Path(f"{config.session_folder()}/cache")
    cacheDir.mkdir(exist_ok = True)

    nodes_to_visualize, color_bar_info, time_invariant_nodes_to_cache, normal_nodoes_to_cache = pipeline(args.i)

    if args.t != -1:

        renderView = create_view()
        renderView.Background = [1.0, 1.0, 1.0]
        renderView.UseColorPaletteForBackground = 0

        for n in nodes_to_visualize:
            props = n.visualize(args.t, renderView)

            if n.id in color_bar_info:
                position, length, title = color_bar_info[n.id]

                lut = props.LookupTable
                bar = GetScalarBar(lut, renderView)

                bar.WindowLocation = "Any Location"
                bar.Position = position
                bar.ScalarBarLength = length
                bar.Title = title
                bar.ComponentTitle = ''

                props.SetScalarBarVisibility(renderView, True)

        if args.o:
            SaveScreenshot(args.o)
        else:
            Interact()

    else:

        for t in range(time_steps):

            renderView = create_view()

            for n in nodes_to_visualize:
                props = n.visualize(t+1, renderView)

                if n.id in color_bar_info:
                    position, length, title = color_bar_info[n.id]

                    lut = props.LookupTable
                    bar = GetScalarBar(lut, renderView)

                    bar.WindowLocation = "Any Location"
                    bar.Position = position
                    bar.ScalarBarLength = length
                    bar.Title = title
                    bar.ComponentTitle = ''

                    props.SetScalarBarVisibility(renderView, True)

            SaveScreenshot(f"{args.q}/frame{str(t+1).zfill(8)}.png", renderView)

            Disconnect()
            tree.clear_evaluation_cache()
            Connect()

        os.system(f"ffmpeg -framerate 4 -i {args.q}/frame%08d.png -c:v libx264 -pix_fmt yuv420p {args.o}")

        for t in range(time_steps):
            os.remove(f"{args.q}/frame{str(t+1).zfill(8)}.png")
