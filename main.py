import os
import numpy as np
from scipy.spatial import Delaunay
import ezdxf
from shapely.geometry import LineString
import pyvista as pv

# Constants
INPUT_DXF = "contours.dxf"
OUTPUT_DXF = "output/lowest_zones.dxf"
TIN_RESOLUTION = 10  # Resolution for TIN grids
INTERSECTION_LAYER = "INTERSECTIONS"  # Single layer for all intersection lines


def read_dxf_contours(file_path):
    """
    Reads contours from a DXF file and returns them grouped by layer.
    Each layer is treated as a separate surface.
    Recognizes both LWPOLYLINE and 3D POLYLINE entities.
    """
    print(f"[DEBUG] Reading DXF file: {file_path}")
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    contours = {}
    found_layers = set()
    found_entities = 0

    for entity in msp:
        found_entities += 1

        # Handle LWPOLYLINE
        if entity.dxftype() == "LWPOLYLINE":
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            if layer_name not in contours:
                contours[layer_name] = []
            elevation = entity.dxf.elevation
            coords = [(p[0], p[1], elevation) for p in entity.get_points()]
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found LWPOLYLINE in layer '{layer_name}' with elevation {elevation} and {len(coords)} points.")

        # Handle 3D POLYLINE
        elif entity.dxftype() == "POLYLINE" and entity.is_3d_polyline:
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            if layer_name not in contours:
                contours[layer_name] = []
            coords = [
                (
                    v.get_dxf_attrib("location").x,
                    v.get_dxf_attrib("location").y,
                    v.get_dxf_attrib("location").z
                )
                for v in entity.vertices
            ]
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found 3D POLYLINE in layer '{layer_name}' with {len(coords)} points.")

        # Handle unsupported entity types
        else:
            print(f"[DEBUG]   Skipped unsupported entity type '{entity.dxftype()}' in layer '{entity.dxf.layer}'.")

    print(f"\n[DEBUG] Total layers in DXF: {len(found_layers)}")
    for layer in found_layers:
        print(f"[DEBUG]   Layer '{layer}' processed.")

    print(f"[DEBUG] Total entities in DXF: {found_entities}")
    print(f"[DEBUG] Total valid contours: {len(contours)}")
    return contours


def generate_tin_from_contours(contours):
    """
    Creates TIN surfaces using contour points for each surface layer.
    Returns a dictionary of TIN meshes.
    """
    tins = {}
    for layer, layer_contours in contours.items():
        print(f"[DEBUG] Generating TIN for layer '{layer}'...")
        points = np.array([point for contour in layer_contours for point in contour])
        if len(points) < 3:
            print(f"[DEBUG]   Warning: Layer '{layer}' has fewer than 3 points, skipping TIN generation.")
            continue

        xy = points[:, :2]  # Use X, Y for triangulation
        tri = Delaunay(xy)
        tins[layer] = {
            "points": points,
            "triangles": tri.simplices,
        }
        print(f"[DEBUG]   TIN generated for layer '{layer}' with {len(tri.simplices)} triangles "
              f"and {len(points)} points.")
    return tins


def visualize_tins(tins, intersections=None):
    """
    Visualizes all TIN surfaces in a single PyVista 3D plot with optional intersections.
    """
    plotter = pv.Plotter()
    for layer, tin in tins.items():
        points = tin["points"]
        triangles = tin["triangles"]

        # Create PyVista mesh
        mesh = pv.PolyData(points)
        mesh.faces = np.hstack([[3] + list(tri) for tri in triangles])
        mesh["Elevation"] = points[:, 2]

        # Add mesh to plotter
        plotter.add_mesh(
            mesh,
            show_edges=True,
            scalars="Elevation",
            cmap="viridis",
            label=layer
        )

    # Visualize intersections in red lines
    if intersections:
        for line in intersections:
            if isinstance(line, LineString):
                coords = np.array(line.coords)
                if coords.shape[0] > 1:
                    # Add the line to the plotter
                    plotter.add_lines(coords, color="red", width=2)

    plotter.show()


def calculate_intersection_lines(tins):
    """
    Calculates 3D intersection lines between TIN surfaces using PyVista's intersection filter.
    Returns a list of shapely LineString objects in 3D.
    """
    print("[DEBUG] Calculating intersection lines between surfaces...")
    intersections = []
    layers = list(tins.keys())

    # Compare each layer with subsequent layers
    for i, layer1 in enumerate(layers):
        for layer2 in layers[i + 1:]:
            print(f"[DEBUG]   Checking intersection between '{layer1}' and '{layer2}'...")
            points1, triangles1 = tins[layer1]["points"], tins[layer1]["triangles"]
            points2, triangles2 = tins[layer2]["points"], tins[layer2]["triangles"]

            # Create PyVista meshes for both TIN surfaces
            mesh1 = pv.PolyData(points1)
            mesh1.faces = np.hstack([[3] + list(tri) for tri in triangles1])
            mesh2 = pv.PolyData(points2)
            mesh2.faces = np.hstack([[3] + list(tri) for tri in triangles2])

            # Calculate intersection lines using PyVista
            # By default, returns (intersection_line, mesh1_split, mesh2_split).
            print(f"[DEBUG]   Performing intersection via PyVista...")
            intersection_line, _, _ = mesh1.intersection(
                mesh2,
                split_first=False,  # We only want the intersection line, not split meshes
                split_second=False
            )

            if isinstance(intersection_line, pv.PolyData):
                print(f"[DEBUG]   Intersection result has {intersection_line.n_points} points "
                      f"and {intersection_line.n_cells} cells.")
                if intersection_line.n_points > 1:
                    # Each cell is typically a "line" in the intersection
                    for cidx in range(intersection_line.n_cells):
                        cell = intersection_line.extract_cells(cidx)
                        if cell.n_points > 1:
                            coords = np.array(cell.points)
                            # Create a 3D shapely LineString
                            line_3d = LineString(coords)
                            intersections.append(line_3d)

    print(f"[DEBUG] Total intersection lines found: {len(intersections)}")
    return intersections


def export_to_dxf(tins, intersections, output_path):
    """
    Exports TIN surfaces and their intersections to a DXF file.
    All intersection lines go to the INTERSECTION_LAYER.
    """
    print("[DEBUG] Exporting results to DXF...")
    doc = ezdxf.new()
    msp = doc.modelspace()

    # 1) Export TIN polygons (projected to 2D as LWPOLYLINE)
    for layer, tin in tins.items():
        points = tin["points"]
        triangles = tin["triangles"]
        print(f"[DEBUG]   Exporting {len(triangles)} triangles for layer '{layer}'...")
        for tri in triangles:
            coords = [(points[i][0], points[i][1]) for i in tri]
            # Close the polyline
            coords.append(coords[0])
            # Add as LWPOLYLINE in this layer
            msp.add_lwpolyline(coords, close=True, dxfattribs={"layer": layer})

    # 2) Export intersection lines to a single layer
    print(f"[DEBUG]   Exporting {len(intersections)} intersection lines to layer '{INTERSECTION_LAYER}'...")
    for idx, line in enumerate(intersections):
        if isinstance(line, LineString):
            coords = list(line.coords)
            # For DXF, we only store XY; Z data is lost in an LWPOLYLINE unless we use 3D POLYLINE
            # If you want 3D in DXF, consider using add_polyline3d
            msp.add_polyline3d(
                coords,
                dxfattribs={"layer": INTERSECTION_LAYER}
            )

    doc.saveas(output_path)
    print(f"[DEBUG] DXF file saved: {output_path}")


def main():
    # Step 1: Read DXF contours
    print("[DEBUG] Step 1: Reading contours from DXF...")
    contours = read_dxf_contours(INPUT_DXF)

    # Step 2: Generate TIN surfaces
    print("\n[DEBUG] Step 2: Generating TIN surfaces...")
    tins = generate_tin_from_contours(contours)

    # Step 3: Calculate intersection lines
    print("\n[DEBUG] Step 3: Calculating intersection lines...")
    intersections = calculate_intersection_lines(tins)

    # Step 4: Visualize TINs and Intersections
    print("\n[DEBUG] Step 4: Visualizing TIN surfaces and intersections...")
    visualize_tins(tins, intersections)

    # Step 5: Export to DXF
    print("\n[DEBUG] Step 5: Exporting results to DXF...")
    os.makedirs(os.path.dirname(OUTPUT_DXF), exist_ok=True)
    export_to_dxf(tins, intersections, OUTPUT_DXF)

    print("\n[DEBUG] Process complete. Results saved to:", OUTPUT_DXF)


if __name__ == "__main__":
    main()
