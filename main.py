import os
import numpy as np
from scipy.spatial import Delaunay
import ezdxf
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
import pyvista as pv

# Constants
INPUT_DXF = "test55.dxf"
OUTPUT_DXF = "output/lowest_zones.dxf"
TIN_RESOLUTION = 10
INTERSECTION_LAYER = "INTERSECTIONS"
SNAP_TOLERANCE = 0.1
DECIMATION_REDUCTION = 0.0

###############################################################################
# STEP 1: READ DXF CONTOURS WITH ENFORCED CLOSURE
###############################################################################

def read_dxf_contours(file_path):
    print(f"[DEBUG] Reading DXF file: {file_path}")
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    contours = {}
    found_layers = set()
    found_entities = 0

    for entity in msp:
        found_entities += 1

        if entity.dxftype() == "LWPOLYLINE":
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            contours.setdefault(layer_name, [])
            elevation = entity.dxf.elevation
            coords = [(p[0], p[1], elevation) for p in entity.get_points()]

            # Enforce closure
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
                print(f"[DEBUG]   Closed LWPOLYLINE in layer '{layer_name}'.")
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found LWPOLYLINE in layer '{layer_name}' with elevation={elevation} and {len(coords)} pts.")

        elif entity.dxftype() == "POLYLINE" and entity.is_3d_polyline:
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            contours.setdefault(layer_name, [])
            coords = [(v.dxf.location.x, v.dxf.location.y, v.dxf.location.z) for v in entity.vertices]

            # Enforce closure
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
                print(f"[DEBUG]   Closed 3D POLYLINE in layer '{layer_name}'.")
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found 3D POLYLINE in layer '{layer_name}' with {len(coords)} pts.")

    print(f"\n[DEBUG] Total layers in DXF: {len(found_layers)} => {found_layers}")
    print(f"[DEBUG] Total entities in DXF: {found_entities}")
    print(f"[DEBUG] Total valid contour layers: {len(contours)}")
    return contours

###############################################################################
# STEP 2: GENERATE TIN FROM CONTOURS
###############################################################################

def generate_tin_from_contours(contours):
    tins = {}
    for layer, layer_contours in contours.items():
        print(f"[DEBUG] Generating TIN for layer '{layer}'...")
        points = np.array([pt for contour_line in layer_contours for pt in contour_line])
        if len(points) < 3:
            print(f"[DEBUG]   Layer '{layer}' has fewer than 3 pts, skipping.")
            continue

        xy = points[:, :2]
        try:
            tri = Delaunay(xy)
            tins[layer] = {"points": points, "triangles": tri.simplices}
            print(f"[DEBUG]   TIN built with {len(tri.simplices)} triangles / {len(points)} pts in layer '{layer}'.")
        except Exception as e:
            print(f"[ERROR]   Failed to build TIN for layer '{layer}': {e}")
    return tins

###############################################################################
# UTILITY: COMPUTE GLOBAL Z-SCALE BASED ON EXTENTS
###############################################################################

def compute_global_z_scale(tins):
    all_points = np.concatenate([tin["points"] for tin in tins.values()])
    xrange = all_points[:,0].max() - all_points[:,0].min()
    yrange = all_points[:,1].max() - all_points[:,1].min()
    zrange = all_points[:,2].max() - all_points[:,2].min()
    planar = max(xrange, yrange)
    scale = planar / zrange if zrange != 0 else 1
    print(f"[DEBUG] Global Z-scale computed: {scale}")
    return scale

###############################################################################
# VISUALIZE TINS WITH CHECKBOXES, AUTO Z-SCALE, LEGENDS, AND INTERSECTIONS
###############################################################################

def visualize_tins_with_checkboxes(tins, intersections=None, global_scale=1.0):
    """
    Visualizes TIN surfaces with checkboxes to toggle visibility, auto Z-scale,
    legends next to checkboxes, and optionally intersection lines.
    """
    print("[DEBUG] Visualizing TIN surfaces with checkboxes and legends...")
    plotter = pv.Plotter()
    actors = {}
    checkbox_positions = {}
    legend_labels = {}
    
    # Determine starting positions for checkboxes and legends
    start_y = 10
    dy = 30
    x_pos = 10
    legend_offset = 25  # horizontal offset for legend text

    for idx, (layer, tin) in enumerate(tins.items()):
        points = tin["points"].copy()
        # Apply global Z-scale uniformly
        points[:,2] *= global_scale  

        triangles = tin["triangles"]
        mesh = pv.PolyData(points)
        mesh.faces = np.hstack([[3] + list(tri) for tri in triangles])
        
        # Explicitly set the "Elevation" array on the mesh's points
        mesh.point_data["Elevation"] = points[:, 2]

        actor = plotter.add_mesh(mesh, show_edges=True, scalars="Elevation", cmap="viridis", label=layer)
        actors[layer] = actor

        # Set position for checkbox widget
        position = (x_pos, start_y + idx * dy)
        checkbox_positions[layer] = position

        # Store legend label position (right to the checkbox)
        legend_labels[layer] = (x_pos + legend_offset, start_y + idx * dy)

    # Add checkboxes and legends
    for layer, actor in actors.items():
        def callback(checked, actor=actor):
            actor.SetVisibility(checked)
        pos = checkbox_positions[layer]
        plotter.add_checkbox_button_widget(callback=callback, value=True, position=pos, size=20)
        legend_pos = legend_labels[layer]
        plotter.add_text(layer, position=legend_pos, font_size=12, color='white', shadow=True)

    # Add intersection lines if provided, using global_scale for Z-axis
    if intersections:
        for line in intersections:
            coords = np.array(line.coords)
            if coords.shape[0] > 1:
                coords[:,2] *= global_scale
                plotter.add_lines(coords, color="red", width=2)

    plotter.add_legend()
    plotter.show()

###############################################################################
# DECIMATE ALL TINS (unchanged)
###############################################################################

def decimate_all_tins(tins, reduction=0.3):
    decimated_tins = {}
    for layer, tin in tins.items():
        points = tin["points"]
        triangles = tin["triangles"]
        mesh = pv.PolyData(points)
        mesh.faces = np.hstack([[3] + list(t) for t in triangles])
        mesh = mesh.clean().triangulate()
        initial_tri_count = mesh.n_cells
        if reduction > 0:
            mesh = mesh.decimate(reduction)
        reduced_tri_count = mesh.n_cells
        print(f"[DEBUG] Layer '{layer}': Triangles reduced from {initial_tri_count} to {reduced_tri_count}")
        new_points = mesh.points
        faces = mesh.faces.reshape((-1, 4))
        new_triangles = faces[:, 1:]
        decimated_tins[layer] = {"points": new_points, "triangles": new_triangles}
    return decimated_tins

###############################################################################
# STEP 3: INTERSECTION LINES (using *DECIMATED* TIN surfaces)
###############################################################################

def calculate_intersection_lines(decimated_tins):
    print("[DEBUG] Calculating intersection lines between decimated TIN surfaces...")
    layers = list(decimated_tins.keys())
    intersections = []
    total_pairs = len(layers) * (len(layers) - 1) // 2
    processed_pairs = 0
    valid_intersections = 0

    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            layer1, layer2 = layers[i], layers[j]
            processed_pairs += 1
            print(f"[DEBUG]   Processing pair {processed_pairs}/{total_pairs}: '{layer1}' vs '{layer2}'")
            pts1 = decimated_tins[layer1]["points"]
            tri1 = decimated_tins[layer1]["triangles"]
            pts2 = decimated_tins[layer2]["points"]
            tri2 = decimated_tins[layer2]["triangles"]

            mesh1 = pv.PolyData(pts1)
            mesh1.faces = np.hstack([[3] + list(t) for t in tri1])
            mesh2 = pv.PolyData(pts2)
            mesh2.faces = np.hstack([[3] + list(t) for t in tri2])

            # Check for collision before attempting intersection
            collision_result = mesh1.collision(mesh2, box_tolerance=1e-5)
            # Unpack collision result: (collision_mesh, n_collisions)
            collision, ncol = collision_result
            if ncol == 0:
                print(f"[DEBUG]     => No collision detected between '{layer1}' and '{layer2}'. Skipping intersection.")
                continue

            try:
                intersection_line, _, _ = mesh1.intersection(mesh2, split_first=False, split_second=False)
                if isinstance(intersection_line, pv.PolyData) and intersection_line.n_points > 1:
                    print(f"[DEBUG]     => Valid intersection found: {intersection_line.n_points} pts, {intersection_line.n_cells} lines.")
                    for cidx in range(intersection_line.n_cells):
                        cell = intersection_line.extract_cells(cidx)
                        if cell.n_points > 1:
                            arr = np.array(cell.points)
                            intersections.append(LineString(arr))
                            valid_intersections += 1
            except Exception as e:
                print(f"[ERROR]     => Failed to compute intersection between '{layer1}' and '{layer2}': {e}")

    print(f"[DEBUG] Total intersection lines found: {valid_intersections}")
    return intersections

###############################################################################
# STEP 3A: SNAP INTERSECTION LINES TO NEAREST ORIGINAL CONTOUR
###############################################################################

def snap_intersections_to_contours(intersections, contours, tolerance=SNAP_TOLERANCE):
    print("[DEBUG] Snapping intersection lines to original contours. Tolerance =", tolerance)
    original_lines = []
    for layer, lines in contours.items():
        for coords in lines:
            original_lines.append(LineString(coords))
    
    snapped = []
    total_snapped = 0
    for line in intersections:
        coords = list(line.coords)
        new_coords = []
        snap_count = 0
        for pt in coords:
            p2d = Point(pt[:2])
            min_d = float("inf")
            best_line = None
            for candidate_line in original_lines:
                d = candidate_line.distance(p2d)
                if d < min_d:
                    min_d = d
                    best_line = candidate_line
            if min_d < tolerance and best_line is not None:
                proj = best_line.interpolate(best_line.project(p2d))
                z_val = pt[2] if len(pt) > 2 else 0
                new_coords.append((proj.x, proj.y, z_val))
                snap_count += 1
            else:
                new_coords.append(pt)
        total_snapped += snap_count
        snapped.append(LineString(new_coords))
    
    print(f"[DEBUG] Total intersection lines processed: {len(intersections)}")
    print(f"[DEBUG] Total snapped points across all intersection lines: {total_snapped}")
    return snapped

###############################################################################
# STEP 4: COMBINE ALL LINES AND POLYGONIZE
###############################################################################

def build_planar_polygons(contours, snapped_intersections):
    print("[DEBUG] Building planar polygons from lines...")

    all_lines_2d = []
    for layer, lines in contours.items():
        for coords in lines:
            coords_2d = [(x, y) for (x, y, z) in coords]
            if coords_2d[0] != coords_2d[-1]:
                coords_2d.append(coords_2d[0])
            all_lines_2d.append(LineString(coords_2d))
    print(f"[DEBUG] Added {len(contours)} sets of lines from contours (with closure enforced).")

    for line in snapped_intersections:
        coords_2d = [(x, y) for (x, y, z) in line.coords]
        if coords_2d[0] != coords_2d[-1]:
            coords_2d.append(coords_2d[0])
        all_lines_2d.append(LineString(coords_2d))
    print(f"[DEBUG] Total lines after adding intersections: {len(all_lines_2d)}")

    visualize_lines(all_lines_2d)

    merged = unary_union(all_lines_2d)
    print(f"[DEBUG] Merged lines type: {type(merged)}")
    if hasattr(merged, '__len__'):
        print(f"[DEBUG] Merged lines count: {len(merged)}")
    else:
        print("[DEBUG] Merged result is a single LineString or similar object.")

    polys = list(polygonize(merged))
    print(f"[DEBUG] Polygonize produced {len(polys)} polygons.")

    for i, poly in enumerate(polys, start=1):
        print(f"[DEBUG] Polygon #{i}: Valid={poly.is_valid}, Closed={poly.is_closed}, Exterior points={len(poly.exterior.coords)}")

    return polys

def visualize_lines(lines_2d):
    print("[DEBUG] Visualizing all input lines before polygonization...")
    plotter = pv.Plotter()
    for line in lines_2d:
        coords = np.array(line.coords)
        coords_3d = np.column_stack((coords, np.zeros(len(coords))))
        if len(coords_3d) > 1:
            polyline = pv.PolyData(coords_3d)
            n = len(coords_3d)
            cells = np.hstack([[n], list(range(n))])
            polyline.lines = cells
            plotter.add_mesh(polyline, color="blue", line_width=2)
    plotter.show()

def visualize_polygons(polygons):
    print("[DEBUG] Visualizing polygon boundaries...")
    plotter = pv.Plotter()
    for poly in polygons:
        coords = np.array(poly.exterior.coords)
        coords_3d = np.column_stack((coords, np.zeros(len(coords))))
        if len(coords_3d) > 1:
            polyline = pv.PolyData(coords_3d)
            n = len(coords_3d)
            cells = np.hstack([[n], list(range(n))])
            polyline.lines = cells
            plotter.add_mesh(polyline, color="green", line_width=3)
    plotter.show()

###############################################################################
# STEP 5: FIND WHICH TIN IS LOWEST FOR EACH POLYGON
###############################################################################

def interpolate_tin_z(x, y, tin):
    points = tin["points"]
    tri = tin["triangles"]
    for simplex in tri:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        x0, y0 = p0[0], p0[1]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
        if abs(denom) < 1e-12:
            continue
        a = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denom
        b = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denom
        c = 1 - a - b
        if (a >= 0) and (b >= 0) and (c >= 0):
            z0, z1, z2 = p0[2], p1[2], p2[2]
            return a*z0 + b*z1 + c*z2
    return None

def label_polygons_with_lowest_tin(polygons, tins):
    print("[DEBUG] Labeling polygons with their lowest TIN surface...")
    labeled = []
    layer_names = list(tins.keys())
    for i, poly in enumerate(polygons, start=1):
        if poly.is_empty:
            continue
        center = poly.centroid
        xC, yC = center.x, center.y
        best_layer = None
        best_z = None
        for layer in layer_names:
            z_val = interpolate_tin_z(xC, yC, tins[layer])
            if z_val is not None:
                if best_z is None or z_val < best_z:
                    best_z = z_val
                    best_layer = layer
        if best_layer is not None:
            labeled.append((poly, best_layer))
        else:
            labeled.append((poly, "NO_TIN"))
        print(f"  Polygon #{i} => lowest = {best_layer}")
    return labeled

###############################################################################
# STEP 6: MERGE ADJACENT POLYGONS WITH THE SAME TIN
###############################################################################

from shapely.ops import unary_union

def merge_polygons_by_tin(labeled_polys):
    print("[DEBUG] Merging adjacent polygons that share the same TIN label...")
    from collections import defaultdict
    grouping = defaultdict(list)
    for poly, layer in labeled_polys:
        grouping[layer].append(poly)
    merged_result = {}
    for layer, polys in grouping.items():
        merged_poly = unary_union(polys)
        merged_result[layer] = merged_poly
        print(f"[DEBUG]   {len(polys)} polygons merged into one geometry for layer '{layer}'")
    return merged_result

###############################################################################
# EXPORT FINAL POLYGONS
###############################################################################

def export_final_polygons(dxf_file, contours, intersection_lines, merged_dict):
    print("[DEBUG] Exporting final results to DXF:", dxf_file)
    doc = ezdxf.new()
    msp = doc.modelspace()

    for layer_name, lines in contours.items():
        for coords in lines:
            msp.add_polyline3d(coords, dxfattribs={"layer": layer_name})

    for i, line in enumerate(intersection_lines, start=1):
        coords = list(line.coords)
        msp.add_polyline3d(coords, dxfattribs={"layer": INTERSECTION_LAYER})

    for layer_label, geom in merged_dict.items():
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            polygons = []
        for poly in polygons:
            if poly.is_empty:
                continue
            xys = list(poly.exterior.coords)
            coords_3d = [(x, y, 0.0) for (x, y) in xys]
            msp.add_polyline3d(coords_3d, dxfattribs={"layer": f"FINAL_{layer_label}"})
            for interior in poly.interiors:
                xys_int = list(interior.coords)
                coords_int_3d = [(x, y, 0.0) for (x, y) in xys_int]
                msp.add_polyline3d(coords_int_3d, dxfattribs={"layer": f"FINAL_{layer_label}_HOLE"})
    doc.saveas(dxf_file)
    print("[DEBUG] DXF export complete.")

###############################################################################
# MAIN WORKFLOW
###############################################################################

def main():
    print("[DEBUG] Step 1: Reading contours...")
    contours = read_dxf_contours(INPUT_DXF)

    print("\n[DEBUG] Step 2: Generating TIN surfaces...")
    tins = generate_tin_from_contours(contours)

    if not tins:
        print("[ERROR] No valid TINs generated. Exiting.")
        return

    # Compute global Z-scale
    global_scale = compute_global_z_scale(tins)

    # Initial visualization with checkboxes
    visualize_tins_with_checkboxes(tins, global_scale=global_scale)

    # Decimate TINs if needed
    decimated_tins = decimate_all_tins(tins, reduction=DECIMATION_REDUCTION)

    print("\n[DEBUG] Step 3: Calculating intersection lines...")
    intersections = calculate_intersection_lines(decimated_tins)

    if not intersections:
        print("[DEBUG] No valid intersection lines found.")
    else:
        print(f"[DEBUG] Total valid intersections after computation: {len(intersections)}")

    print("\n[DEBUG] Step 3a: Snapping intersection lines to original contours...")
    snapped_intersections = snap_intersections_to_contours(intersections, contours, tolerance=SNAP_TOLERANCE)

    print("\n[DEBUG] Step 4: Building planar polygons from lines (original + intersection)...")
    polygons = build_planar_polygons(contours, snapped_intersections)

    if not polygons:
        print("[DEBUG] No polygons were created. Exiting.")
        return

    visualize_polygons(polygons)

    print("\n[DEBUG] Step 5: Label each polygon with its lowest TIN surface...")
    labeled_polys = label_polygons_with_lowest_tin(polygons, decimated_tins)

    print("\n[DEBUG] Step 6: Merge polygons by TIN label...")
    merged_by_tin = merge_polygons_by_tin(labeled_polys)

    print("\n[DEBUG] Visualizing final TINs with intersections, checkboxes, and legends...")
    visualize_tins_with_checkboxes(decimated_tins, intersections=snapped_intersections, global_scale=global_scale)

    print("\n[DEBUG] Step 7: Export results to DXF...")
    os.makedirs(os.path.dirname(OUTPUT_DXF), exist_ok=True)
    export_final_polygons(OUTPUT_DXF, contours, snapped_intersections, merged_by_tin)

    print(f"\n[DEBUG] Workflow complete. Results saved to '{OUTPUT_DXF}'.")

if __name__ == "__main__":
    main()
