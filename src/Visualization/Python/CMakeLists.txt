# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  InterpolateVolumeData.py
  PYTHON_EXECUTABLES
  GenerateXdmf.py
  PlotDatFile.py
  Render1D.py
)
