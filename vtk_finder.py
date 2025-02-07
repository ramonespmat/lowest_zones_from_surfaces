import os
import glob
import vtk

# Get the VTK package folder
vtk_folder = os.path.dirname(vtk.__file__)
print("VTK package folder:", vtk_folder)

# Search recursively for vtkCommonCore (could be .pyd or .dll)
pattern = os.path.join(vtk_folder, "**", "vtkCommonCore*")
matches = glob.glob(pattern, recursive=True)

if matches:
    print("Found the following vtkCommonCore files:")
    for m in matches:
        print("  ", m)
else:
    print("No vtkCommonCore files found with pattern:", pattern)
