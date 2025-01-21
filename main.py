import os
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import ezdxf
from shapely.geometry import LineString, Point
import pyvista as pv

# Constants
INPUT_DXF = "contours_extra.dxf"
OUTPUT_DXF = "output/lowest_zones.dxf"
TIN_RESOLUTION = 10  # Resolution for TIN grids
INTERSECTION_LAYER = "INTERSECTIONS"  # Single layer for all intersection lines
SNAP_TOLERANCE = 0.1  # Snap if intersection vertex is within 1 cm (for example)


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

        # Handle LWPOLYLINE (usually 2D with an elevation)
        if entity.dxftype() == "LWPOLYLINE":
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            if layer_name not in contours:
                contours[layer_name] = []
            elevation = entity.dxf.elevation
            coords = [(p[0], p[1], elevation) for p in entity.get_points()]
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found LWPOLYLINE in layer '{layer_name}' with elevation {elevation} "
                  f"and {len(coords)} points.")

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

        else:
            # Skipped unsupported entity types
            pass

    print(f"\n[DEBUG] Total layers in DXF: {len(found_layers)}")
    for layer in found_layers:
        print(f"[DEBUG]   Layer '{layer}' processed.")

    print(f"[DEBUG] Total entities in DXF: {found_entities}")
    print(f"[DEBUG] Total valid contours: {len(contours)}")
    return contours


def generate_tin_from_contours(contours):
    """
    Creates TIN surfaces using contour points for each surface layer.
    Returns a dictionary of TIN meshes (for intersection computation).
    """
    tins = {}
    for layer, layer_contours in contours.items():
        print(f"[DEBUG] Generating TIN for layer '{layer}'...")
        # Flatten all lines within this layer into a single point array
        points = np.array([point for contour_line in layer_contours for point in contour_line])
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
            intersection_line, _, _ = mesh1.intersection(
                mesh2,
                split_first=False,   # Only the intersection line
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


def snap_intersections_to_contours(
    intersections,
    contours,
    tolerance=SNAP_TOLERANCE
):
    """
    For each vertex in each intersection line, snap it to the nearest point 
    on the closest original contour line if within the given tolerance.

    :param intersections: List of shapely LineString objects.
    :param contours: Dictionary of layer -> list_of_lines -> coordinates.
    :param tolerance: Maximum distance to perform snapping.
    :return: A new list of LineStrings with snapped coordinates.
    """
    print("[DEBUG] Starting snapping process using manual nearest line search...")
    print(f"[DEBUG]   Tolerance: {tolerance}")

    # 1) Build a list of all original contour lines as Shapely LineString objects.
    original_lines = []
    for layer_name, lines in contours.items():
        for line_coords in lines:
            # Convert each contour line to a Shapely LineString.
            original_lines.append(LineString(line_coords))
    print(f"[DEBUG]   Total original contour lines: {len(original_lines)}")

    snapped_intersections = []
    total_snap_count = 0
    line_index = 0

    # 2) Process each intersection line.
    for line in intersections:
        line_index += 1
        coords = list(line.coords)
        new_coords = []
        snap_count_line = 0

        print(f"[DEBUG] - Processing intersection line #{line_index} with {len(coords)} points.")
        # 3) For each vertex in the intersection line:
        for j, pt in enumerate(coords):
            # Create a Shapely point from the current vertex.
            point = Point(pt[:2])  # Use X, Y for projection.

            # 4) Manually find the nearest original line.
            min_dist = float('inf')
            nearest_line = None
            for candidate in original_lines:
                d = point.distance(candidate)
                if d < min_dist:
                    min_dist = d
                    nearest_line = candidate
            distance = min_dist

            # 5) If within tolerance, project the point onto the nearest line.
            if distance < tolerance and nearest_line is not None:
                projected = nearest_line.interpolate(nearest_line.project(point))
                # Preserve original Z if available.
                z_val = pt[2] if len(pt) > 2 else 0.0
                snapped_pt = (projected.x, projected.y, z_val)
                new_coords.append(snapped_pt)
                snap_count_line += 1
                print(f"[DEBUG]   Snapped point {j} from {pt} to {snapped_pt} (dist={distance:.6f})")
            else:
                new_coords.append(pt)
                print(f"[DEBUG]   Point {j} not snapped, distance {distance:.6f} exceeds tolerance.")

        total_snap_count += snap_count_line
        print(f"[DEBUG] - End of line #{line_index}, snapped {snap_count_line} points.")

        # 6) Create a new LineString with the snapped coordinates.
        snapped_intersections.append(LineString(new_coords))

    print(f"[DEBUG] Finished snapping. Total snapped points across all lines: {total_snap_count}")
    return snapped_intersections



def visualize_tins(tins, intersections=None):
    """
    Visualizes TIN surfaces in 3D plus optional intersection lines.
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

    if intersections:
        for line in intersections:
            coords = np.array(line.coords)
            if coords.shape[0] > 1:
                plotter.add_lines(coords, color="red", width=2)

    plotter.show()


def export_to_dxf(contours, intersections, output_path):
    """
    Exports the original contour polylines (as read) plus the intersection lines
    into a single DXF file.
    """
    print("[DEBUG] Exporting results to DXF...")
    doc = ezdxf.new()
    msp = doc.modelspace()

    # 1) Export the original polylines exactly as read
    print(f"[DEBUG]   Exporting original contour lines to DXF...")
    for layer_name, list_of_lines in contours.items():
        for line_coords in list_of_lines:
            # 3D polylines for original
            msp.add_polyline3d(
                line_coords,
                dxfattribs={"layer": layer_name}
            )

    # 2) Export intersection lines to a single layer
    print(f"[DEBUG]   Exporting {len(intersections)} intersection lines to layer '{INTERSECTION_LAYER}'...")
    for idx, line in enumerate(intersections):
        coords = list(line.coords)
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

    # Step 2: Generate TIN surfaces (for intersection, but not re-exported)
    print("\n[DEBUG] Step 2: Generating TIN surfaces...")
    tins = generate_tin_from_contours(contours)

    # Step 3: Calculate intersection lines
    print("\n[DEBUG] Step 3: Calculating intersection lines...")
    intersections = calculate_intersection_lines(tins)

    # Step 3a: Snap intersections to original contours
    print("\n[DEBUG] Step 3a: Snapping intersection lines to original contours...")
    snapped_intersections = snap_intersections_to_contours(
        intersections, 
        contours, 
        tolerance=SNAP_TOLERANCE
    )

    # Step 4: Visualize TINs and Intersections (optional)
    print("\n[DEBUG] Step 4: Visualizing TIN surfaces and (snapped) intersections...")
    visualize_tins(tins, snapped_intersections)

    # Step 5: Export to DXF (original lines + intersection lines)
    print("\n[DEBUG] Step 5: Exporting results to DXF...")
    os.makedirs(os.path.dirname(OUTPUT_DXF), exist_ok=True)
    export_to_dxf(contours, snapped_intersections, OUTPUT_DXF)

    print("\n[DEBUG] Process complete. Results saved to:", OUTPUT_DXF)


if __name__ == "__main__":
    main()
