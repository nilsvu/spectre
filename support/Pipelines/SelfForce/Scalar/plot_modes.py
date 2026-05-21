import paraview.simple as pv
from pathlib import Path
import numpy as np
import subprocess
import yaml


spin = -0.998

base_dir = Path("/Users/nilsvu/Projects/spectre/build-Default-Release/test_ssf71/r_isco")
spin_dir = base_dir / f"a{spin}"
with open(spin_dir / "m0/ScalarSelfForce.yaml") as f:
    _, input_file = yaml.safe_load_all(f)
    block_bounds = input_file["DomainCreator"]["AlignedLattice"]["BlockBounds"][0]
    orbital_radius = (block_bounds[2] + block_bounds[3]) / 2
    inner_radius = block_bounds[0]
# clip_radius = orbital_radius + 2.0 * (orbital_radius - inner_radius)
clip_radius = np.ceil(2 * orbital_radius)

radial_scale = 1.0 / orbital_radius
axis_color = [0.2] * 3
axis_font_size = 32
num_colored_modes = 4

# paraview.simple._DisableFirstRenderCameraReset()
renderView = pv.GetActiveViewOrCreate('RenderView')
renderView.UseColorPaletteForBackground = 0
renderView.Background = [1.0, 1.0, 1.0]
renderView.OrientationAxesVisibility = 0
renderView.AxesGrid.Visibility = 1
renderView.AxesGrid.Set(
    # XTitle="$r$",
    # YTitle="$z^2$",
    # ZTitle=r"$\Psi$",
    XTitle="",
    YTitle="",
    ZTitle="",
    XLabelColor=axis_color,
    YLabelColor=axis_color,
    ZLabelColor=axis_color,
    XTitleColor=axis_color,
    YTitleColor=axis_color,
    ZTitleColor=axis_color,
    XLabelFontSize=axis_font_size,
    YLabelFontSize=axis_font_size,
    ZLabelFontSize=axis_font_size,
    XTitleFontSize=axis_font_size,
    YTitleFontSize=axis_font_size,
    ZTitleFontSize=axis_font_size,
    XTitleFontFamily='Times',
    YTitleFontFamily='Times',
    ZTitleFontFamily='Times',
    XLabelFontFamily='Times',
    YLabelFontFamily='Times',
    ZLabelFontFamily='Times',
    GridColor=axis_color,
    DataScale=[radial_scale, 1.0, 0.5],
    XAxisUseCustomLabels=0,
    # XAxisLabels=block_bounds,
    YAxisUseCustomLabels=1,
    YAxisLabels=[0, 0.5, 1],
    AxesToLabel = 30,
    ZAxisUseCustomLabels=1,
    ZAxisLabels=list(range(-5, 1)),
    UseCustomBounds=1,
    CustomBounds=[inner_radius * radial_scale, clip_radius * radial_scale, 0.0, 1.0, -3, 0.0],
)

def show(filename, color):
    data = pv.XDMFReader(FileNames=[filename])
    calculator = pv.Calculator(Input=data)
    calculator.Function = 'log10(sqrt("Re(MMode)"^2+"Im(MMode)"^2))'
    warpByScalar = pv.WarpByScalar(Input=calculator)
    warpByScalar.Scalars = ['POINTS', 'Result']
    warpByScalar.ScaleFactor = 0.5
    clip_z = pv.Clip(Input=warpByScalar)
    clip_z.ClipType.Origin = [0.0, 0.0, -3]
    clip_z.ClipType.Normal = [0.0, 0.0, -1.0]
    clip = pv.Clip(Input=clip_z)
    clip.ClipType.Origin = [clip_radius, 0.0, 0.0]
    clipDisplay = pv.Show(clip, renderView, 'UnstructuredGridRepresentation')
    clipDisplay.SetRepresentationType('Wireframe')
    clipDisplay.Scale = [radial_scale, 1.0, 1.0]
    clipDisplay.DataAxesGrid.Scale = [radial_scale, 1.0, 1.0]
    clipDisplay.PolarAxes.Scale = [radial_scale, 1.0, 1.0]
    clipDisplay.Opacity = 0.6 if m <= num_colored_modes else (1 - m / 23) * 0.6
    pv.ColorBy(clipDisplay, None)
    clipDisplay.Set(AmbientColor=color, DiffuseColor=color)

colors = np.array([
    [192, 57, 43],
    [241, 196, 15],
    [26, 188, 156],
    [41, 128, 185],
    [231, 76, 60],
    [243, 156, 18],
    [155, 89, 182],
    [39, 174, 96],
    [230, 126, 34],
]) / 255

for m in range(21):
    run_dir = spin_dir / f"m{m}"
    if not (run_dir / "ssf.xmf").exists():
        subprocess.run(["/Users/nilsvu/Projects/spectre/build-Default-Release/bin/spectre", "generate-xdmf", run_dir / "ScalarSelfForceVolume0.h5", "-o", run_dir / "ssf.xmf"])
    # color = colors[m % len(colors)]
    color = colors[m] if m <= num_colored_modes else 3 * [m / 23]
    show(str(run_dir / "ssf.xmf"), color=color)

animationScene = pv.GetAnimationScene()
animationScene.GoToLast()

renderView.Set(
    CameraPosition=[1, -6, 1],
    CameraFocalPoint=[1, 0, -1.25],
    CameraViewUp=[0, 0, 1],
)

layout = pv.GetLayout()
height = 800
layout.SetSize(int(height * 0.75), height)


pv.SaveScreenshot(f"ssf_modes_a{spin}.png")
