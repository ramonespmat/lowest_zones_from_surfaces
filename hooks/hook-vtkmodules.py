from PyInstaller.utils.hooks import collect_dynamic_libs
import os
import vtk

# For vtkmodules (the .pyd files)
binaries = collect_dynamic_libs("vtkmodules")

# Additionally, manually add binaries from vtk.libs
vtk_libs_folder = os.path.join(os.path.dirname(vtk.__file__), "vtk.libs")
if os.path.isdir(vtk_libs_folder):
    import glob
    for dll in glob.glob(os.path.join(vtk_libs_folder, "*.dll")):
        binaries.append((dll, "vtk.libs"))
