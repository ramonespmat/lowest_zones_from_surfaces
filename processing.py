# processing.py

import os
import random
import numpy as np
from scipy.spatial import Delaunay
import ezdxf
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union, split, snap
from shapely.prepared import prep
import pyvista as pv
from collections import defaultdict
import threading  # For running visualization in a separate thread

# Constants
INTERSECTION_LAYER = "INTERSECTIONS"

def read_dxf_contours(file_path, progress_callback=None):
    if progress_callback:
        progress_callback(f"STEP: Reading DXF file: {file_path}")
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

def generate_tin_from_contours(contours, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Generating TIN surfaces...")
    tins = {}
    for layer, layer_contours in contours.items():
        if progress_callback:
            progress_callback(f"Generating TIN for layer '{layer}'...")
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

def compute_global_z_scale(tins, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Computing global Z-scale based on extents...")
    all_points = np.concatenate([tin["points"] for tin in tins.values()])
    xrange = all_points[:,0].max() - all_points[:,0].min()
    yrange = all_points[:,1].max() - all_points[:,1].min()
    zrange = all_points[:,2].max() - all_points[:,2].min()
    planar = max(xrange, yrange)
    scale = planar / zrange if zrange != 0 else 1
    print(f"[DEBUG] Global Z-scale computed: {scale}")
    return scale

def visualize_tins_with_checkboxes(tins, intersections=None, global_scale=1.0, progress_callback=None):
    print("[DEBUG] Visualizing TIN surfaces with checkboxes...")
    plotter = pv.Plotter(lighting='light_kit')  # Explicit lighting setup

    # Positioning parameters
    start_y, dy, x_pos = 10, 30, 10

    for idx, (layer, tin) in enumerate(tins.items()):
        # Create mesh with original lighting parameters
        points = tin["points"].copy()
        points[:, 2] *= global_scale
        mesh = pv.PolyData(points, faces=np.hstack([[3] + list(tri) for tri in tin["triangles"]]))
        mesh.point_data["Elevation"] = points[:, 2]

        # Add mesh with original visualization settings
        y_pos = start_y + idx * dy
        actor = plotter.add_mesh(
            mesh,
            show_edges=False,
            scalars="Elevation",
            cmap="viridis",
            label=layer
        )

        # Add checkbox widget
        plotter.add_checkbox_button_widget(
            lambda state, a=actor: a.SetVisibility(state),
            value=True,
            position=(x_pos, y_pos),
            size=20
        )
        plotter.add_text(layer, position=(x_pos + 25, y_pos), font_size=12)

    # Add intersections with proper lighting
    if intersections:
        all_lines = []
        for line in intersections:
            reduced = reduce_lines_in_intersection(line)
            coords = np.array(reduced.coords)
            coords[:, 2] *= global_scale
            all_lines.append(coords)

        if all_lines:
            plotter.add_lines(
                np.concatenate(all_lines), 
                color="red", 
                width=2,
            )

    plotter.add_legend()
    plotter.show()

def reduce_lines_in_intersection(line, max_points=200):
    """Ultra-fast point reduction with stride-based sampling"""
    coords = np.array(line.coords)
    if len(coords) <= max_points:
        return line
    
    # Calculate optimal stride for fastest access pattern
    stride = max(1, len(coords) // max_points)
    reduced_coords = coords[::stride]
    
    # Ensure we don't exceed max_points while maintaining endpoints
    if len(reduced_coords) > max_points:
        reduced_coords = np.concatenate([reduced_coords[:max_points//2], reduced_coords[-max_points//2:]])
    
    print(f"Reduced {len(coords)} -> {len(reduced_coords)} points (stride {stride})")
    return LineString(reduced_coords)

def decimate_all_tins(tins, reduction=0.3, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Decimating TIN surfaces...")
    decimated_tins = {}
    for layer, tin in tins.items():
        points = tin["points"]
        triangles = tin["triangles"]

        # Skip decimation if triangles are too few
        if len(triangles) <= 20:
            print(f"[DEBUG] Layer '{layer}': Skipping decimation (only {len(triangles)} triangles).")
            decimated_tins[layer] = tin
            continue

        # Create PolyData for processing
        mesh = pv.PolyData(points)
        mesh.faces = np.hstack([[3] + list(t) for t in triangles])
        mesh = mesh.clean().triangulate()  # Pre-cleaning step
        initial_tri_count = mesh.n_cells

        # Dynamically adjust reduction factor based on triangle count
        if initial_tri_count > 1000:
            local_reduction = 0.6
        elif initial_tri_count > 500:
            local_reduction = 0.4
        else:
            local_reduction = 0.2

        print(f"[DEBUG] Layer '{layer}': Using reduction = {local_reduction}.")
        
        # Perform decimation without `preserve_topology`
        try:
            if local_reduction > 0:
                mesh = mesh.decimate(local_reduction)
            reduced_tri_count = mesh.n_cells

            # Handle cases where decimation doesn't reduce significantly
            if reduced_tri_count == initial_tri_count:
                print(f"[DEBUG] Layer '{layer}': Decimation did not reduce triangles significantly.")

            print(f"[DEBUG] Layer '{layer}': Triangles reduced from {initial_tri_count} to {reduced_tri_count}.")
            new_points = mesh.points
            faces = mesh.faces.reshape((-1, 4))
            new_triangles = faces[:, 1:]
            decimated_tins[layer] = {"points": new_points, "triangles": new_triangles}

        except Exception as e:
            print(f"[ERROR] Layer '{layer}': Decimation failed: {e}")
            decimated_tins[layer] = tin  # Add the original mesh if decimation fails

    return decimated_tins

def calculate_intersection_lines(decimated_tins, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Calculating intersection lines between TIN surfaces...")
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
            if progress_callback:
                progress_callback(f"Processing pair {processed_pairs}/{total_pairs}: '{layer1}' vs '{layer2}'")
            print(f"[DEBUG]   Processing pair {processed_pairs}/{total_pairs}: '{layer1}' vs '{layer2}'")
            pts1 = decimated_tins[layer1]["points"]
            tri1 = decimated_tins[layer1]["triangles"]
            pts2 = decimated_tins[layer2]["points"]
            tri2 = decimated_tins[layer2]["triangles"]

            mesh1 = pv.PolyData(pts1)
            mesh1.faces = np.hstack([[3] + list(t) for t in tri1])
            
            mesh2 = pv.PolyData(pts2)
            mesh2.faces = np.hstack([[3] + list(t) for t in tri2])
            # Collision check
            collision_result = mesh1.collision(mesh2, box_tolerance=1e-0)
            collision, ncol = collision_result
            if ncol == 0:
                print(f"[DEBUG]     => No collision detected between '{layer1}' and '{layer2}'. Skipping intersection.")
                continue

            # Intersection
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

def snap_intersections_to_contours(intersections, contours, tolerance=0.1, progress_callback=None):
    """
    Snaps each intersection line's vertices to the nearest contour geometry if
    within tolerance.
    """
    if progress_callback:
        progress_callback("STEP: Snapping intersection lines to original contours...")
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

def split_lines_at_crossings_with_strtree(lines_2d, progress_callback=None):
    """
    Splits lines at all crossing points using STRtree for efficiency.
    """
    if progress_callback:
        progress_callback("Splitting lines at crossing points...")
    print("[DEBUG] Splitting lines at crossing points with STRtree...")

    changed = True
    iteration_count = 0

    geoms_list = lines_2d[:]  # make a copy
    geoms_map = {id(g): i for i, g in enumerate(geoms_list)}

    while changed:
        changed = False
        iteration_count += 1

        tree = STRtree(geoms_list)

        new_geoms = []
        used = set()

        n = len(geoms_list)
        print(f"[DEBUG]   pass={iteration_count}, building STRtree with {n} lines.")

        # Print progress every 10% of lines
        progress_step = max(n // 10, 1)

        for i, line in enumerate(geoms_list):
            if i % progress_step == 0:
                pct = (i / n) * 100
                print(f"[DEBUG]     progress: {i}/{n} (~{pct:.0f}%) in pass={iteration_count}")
                if progress_callback:
                    progress_callback(f"Splitting lines: pass {iteration_count}, {pct:.0f}% complete")

            if i in used:
                continue

            splitted_i = [line]
            replaced_i = False

            candidates = tree.query(line)

            for candidate in candidates:
                if candidate is line:
                    continue

                cand_id = id(candidate)
                j = geoms_map.get(cand_id, None)
                if j is None:
                    continue

                if j in used:
                    continue

                inter = splitted_i[-1].intersection(candidate)
                if not inter.is_empty and inter.geom_type == 'Point':
                    pt = inter
                    if (not pt.equals(Point(splitted_i[-1].coords[0])) and
                        not pt.equals(Point(splitted_i[-1].coords[-1])) and
                        not pt.equals(Point(candidate.coords[0])) and
                        not pt.equals(Point(candidate.coords[-1]))):
                        
                        splitted1 = split(splitted_i.pop(), pt)
                        splitted_i.extend(list(splitted1.geoms))

                        splitted2 = split(candidate, pt)

                        used.add(j)

                        new_geoms.extend(list(splitted2.geoms))

                        replaced_i = True
                        changed = True

            if replaced_i:
                used.add(i)
                new_geoms.extend(splitted_i)
            else:
                new_geoms.append(line)

        geoms_list = new_geoms
        geoms_map = {id(g): idx for idx, g in enumerate(geoms_list)}

        print(f"[DEBUG]   pass={iteration_count} => total lines now = {len(geoms_list)}")
        if progress_callback:
            progress_callback(f"Splitting lines: pass {iteration_count} complete")

    print("[DEBUG] Done splitting lines with STRtree.")
    return geoms_list

def snap_all_lines(lines_2d, tolerance):
    """
    Snaps a list of LineStrings to each other so that near misses become real shared vertices within tolerance.
    """
    merged = unary_union(lines_2d)
    snapped_merged = snap(merged, merged, tolerance)
    if snapped_merged.is_empty:
        return []

    out_lines = []
    if snapped_merged.geom_type == 'LineString':
        out_lines.append(snapped_merged)
    elif snapped_merged.geom_type == 'MultiLineString':
        out_lines.extend(list(snapped_merged.geoms))
    else:
        for g in snapped_merged.geoms:
            if g.geom_type == 'LineString':
                out_lines.append(g)
            elif g.geom_type == 'MultiLineString':
                out_lines.extend(list(g.geoms))
    return out_lines

def build_planar_polygons(contours, snapped_intersections, snap_tolerance=0.1, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Building planar polygons from lines...")
    print("[DEBUG] Building planar polygons from lines...")

    # 1) Convert all contour lines to 2D
    contour_lines_2d = []
    for layer, lines in contours.items():
        for coords in lines:
            coords_2d = [(x, y) for (x, y, z) in coords]
            if coords_2d and coords_2d[0] != coords_2d[-1]:
                coords_2d.append(coords_2d[0])
            contour_lines_2d.append(LineString(coords_2d))

    # 2) Convert intersection lines to 2D
    intersection_lines_2d = []
    for line in snapped_intersections:
        coords_2d = [(x, y) for (x, y, z) in line.coords]
        if coords_2d and coords_2d[0] != coords_2d[-1]:
            coords_2d.append(coords_2d[0])
        intersection_lines_2d.append(LineString(coords_2d))

    # Combine them
    all_lines_2d = contour_lines_2d + intersection_lines_2d
    print(f"[DEBUG] Total lines before snapping: {len(all_lines_2d)}")
    if progress_callback:
        progress_callback(f"Total lines before snapping: {len(all_lines_2d)}")

    # 3) Snap all lines so near misses become real intersections
    snapped_lines_2d = snap_all_lines(all_lines_2d, snap_tolerance)
    print("[DEBUG] Lines snapped to each other with tolerance =", snap_tolerance)
    if progress_callback:
        progress_callback("Snapping lines to each other completed.")

    # 4) Split lines at crossing points
    splitted_lines_2d = split_lines_at_crossings_with_strtree(snapped_lines_2d, progress_callback)
    print(f"[DEBUG] => After splitting, we have {len(splitted_lines_2d)} lines total.")
    if progress_callback:
        progress_callback(f"After splitting, {len(splitted_lines_2d)} lines remain.")

    # 5) Merge lines
    merged = unary_union(splitted_lines_2d)

    # 6) Polygonize
    polys = list(polygonize(merged))
    print(f"[DEBUG] => Polygonize produced {len(polys)} polygons.")
    if progress_callback:
        progress_callback(f"Polygonization complete: {len(polys)} polygons created.")

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

def label_polygons_with_lowest_tin(polygons, tins, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Labeling polygons with their lowest TIN surface...")
    print("[DEBUG] Labeling polygons with their lowest TIN surface...")
    labeled = []
    layer_names = list(tins.keys())

    # Precompute bounding boxes for each TIN
    tin_bboxes = {}
    for layer in layer_names:
        pts = tins[layer]["points"]
        x_min, x_max = pts[:,0].min(), pts[:,0].max()
        y_min, y_max = pts[:,1].min(), pts[:,1].max()
        tin_bboxes[layer] = (x_min, x_max, y_min, y_max)

    total_polygons = len(polygons)
    for i, poly in enumerate(polygons, start=1):
        if poly.is_empty:
            labeled.append((poly, "NO_TIN"))
            continue

        # Use a point guaranteed inside
        rep_pt = poly.representative_point()
        xC, yC = rep_pt.x, rep_pt.y

        best_layer = None
        best_z = None

        for layer in layer_names:
            (x_min, x_max, y_min, y_max) = tin_bboxes[layer]
            if not (x_min <= xC <= x_max and y_min <= yC <= y_max):
                continue  # skip TIN because it's definitely out of XY range

            z_val = interpolate_tin_z(xC, yC, tins[layer])
            if z_val is not None:
                if best_z is None or z_val < best_z:
                    best_z = z_val
                    best_layer = layer

        if best_layer is not None:
            labeled.append((poly, best_layer))
        else:
            labeled.append((poly, "NO_TIN"))

        print(f"  Polygon #{i} => sample=({xC:.2f},{yC:.2f}); assigned={best_layer}")
        if progress_callback and i % max(1, total_polygons//10) == 0:
            progress_callback(f"Labeling polygons: {i}/{total_polygons} done")

    return labeled

def visualize_labeled_polygons(labeled_polys):
    print("[DEBUG] Visualizing labeled polygons...")
    plotter = pv.Plotter()
    unique_labels = list(set(layer for poly, layer in labeled_polys))
    colors = {label: [random.random(), random.random(), random.random()] for label in unique_labels}
    for poly, layer in labeled_polys:
        coords = np.array(poly.exterior.coords)
        coords_3d = np.column_stack((coords, np.zeros(len(coords))))
        if len(coords_3d) > 1:
            polydata = pv.PolyData(coords_3d)
            n = len(coords_3d)
            cells = np.hstack([[n], list(range(n))])
            polydata.lines = cells
            color = colors.get(layer, [0.5, 0.5, 0.5])
            plotter.add_mesh(polydata, color=color, line_width=3)
            centroid = poly.centroid
            label_point = np.array([[centroid.x, centroid.y, 0.0]])
            plotter.add_point_labels(label_point, [layer], text_color='white', font_size=10)
    plotter.show()

def merge_polygons_by_tin(labeled_polys, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Merging adjacent polygons with the same TIN label...")
    print("[DEBUG] Merging adjacent polygons that share the same TIN label...")
    grouping = defaultdict(list)
    for poly, layer in labeled_polys:
        grouping[layer].append(poly)
    merged_result = {}
    for layer, polys in grouping.items():
        merged_poly = unary_union(polys)
        merged_result[layer] = merged_poly
        print(f"[DEBUG]   {len(polys)} polygons merged into one geometry for layer '{layer}'")
        if progress_callback:
            progress_callback(f"Merged {len(polys)} polygons into '{layer}'")
    return merged_result

def export_final_polygons(dxf_file, contours, intersection_lines, merged_dict, progress_callback=None):
    if progress_callback:
        progress_callback("STEP: Exporting final results to DXF...")
    print("[DEBUG] Exporting final results to DXF:", dxf_file)
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Export original contour lines
    for layer_name, lines in contours.items():
        for coords in lines:
            msp.add_polyline3d(coords, dxfattribs={"layer": layer_name})

    # Export all computed intersection lines to the INTERSECTION_LAYER
    for i, line in enumerate(intersection_lines, start=1):
        coords = list(line.coords)
        msp.add_polyline3d(coords, dxfattribs={"layer": INTERSECTION_LAYER})

    # ---- NEW: Export used intersections ----
    print("[DEBUG] Exporting used intersections to 'USED_INTERSECTIONS' layer...")
    used_intersections = filter_used_intersections(intersection_lines, merged_dict, tolerance=0.1)
    total_used = len(used_intersections)
    for i, line in enumerate(used_intersections, start=1):
        coords = list(line.coords)
        msp.add_polyline3d(coords, dxfattribs={"layer": "USED_INTERSECTIONS"})
        # Print progress every 10%
        if i % max(1, total_used // 10) == 0:
            print(f"[DEBUG] Exported {i}/{total_used} used intersections to DXF.")
    # ---- End New Section ----

    # Export merged polygons by TIN label
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
    if progress_callback:
        progress_callback("DXF export complete.")



def find_lowest_surfaces(input_dxf, output_dxf, snap_tolerance=0.1, decimation_reduction=0.5, visualize=False, progress_callback=None):
    """
    Main function to process the DXF and find lowest surfaces.

    Parameters:
    - input_dxf (str): Path to the input DXF file.
    - output_dxf (str): Path to save the output DXF file.
    - snap_tolerance (float): Tolerance for snapping intersection lines.
    - decimation_reduction (float): Decimation reduction factor.
    - visualize (bool): Whether to visualize the intersections in 3D.
    - progress_callback (callable): Function to call with progress messages.
    
    Returns:
    - merged_by_tin (dict): Merged polygons by TIN label.
    """
    print("[DEBUG] Step 1: Reading contours...")
    contours = read_dxf_contours(input_dxf, progress_callback)
    if not contours:
        print("[ERROR] No contours found. Exiting.")
        if progress_callback:
            progress_callback("No contours found. Exiting.")
        return {}

    print("\n[DEBUG] Step 2: Generating TIN surfaces...")
    tins = generate_tin_from_contours(contours, progress_callback)
    if not tins:
        print("[ERROR] No valid TINs generated. Exiting.")
        if progress_callback:
            progress_callback("No valid TINs generated. Exiting.")
        return {}

    # Compute global Z-scale
    global_scale = compute_global_z_scale(tins, progress_callback)

    # (Optional) visualize original TINs
    # visualize_tins_with_checkboxes(tins, global_scale=global_scale)

    print("\n[DEBUG] Step 2A: Decimating TINs for intersection")
    if progress_callback:
        progress_callback("Decimating TIN surfaces...")
    decimated_tins = decimate_all_tins(tins, reduction=decimation_reduction, progress_callback=progress_callback)

    print("\n[DEBUG] Step 3: Calculating intersection lines...")
    intersections = calculate_intersection_lines(decimated_tins, progress_callback)
    if not intersections:
        print("[DEBUG] No valid intersection lines found.")
        if progress_callback:
            progress_callback("No valid intersection lines found.")
    else:
        print(f"[DEBUG] Total valid intersections after computation: {len(intersections)}")
        if progress_callback:
            progress_callback(f"Total valid intersections found: {len(intersections)}")

    print("\n[DEBUG] Step 3a: Snapping intersection lines to original contours...")
    snapped_intersections = snap_intersections_to_contours(intersections, contours, tolerance=snap_tolerance, progress_callback=progress_callback)

    # Optional Visualization after snapping intersections
    if visualize:
        def visualize_thread():
            visualize_tins_with_checkboxes(decimated_tins, snapped_intersections, global_scale=global_scale)
        threading.Thread(target=visualize_thread, daemon=True).start()

    print("\n[DEBUG] Step 4: Building planar polygons from lines (contours + intersections)...")
    polygons = build_planar_polygons(contours, snapped_intersections, snap_tolerance=snap_tolerance, progress_callback=progress_callback)
    if not polygons:
        print("[DEBUG] No polygons were created. Exiting.")
        if progress_callback:
            progress_callback("No polygons were created. Exiting.")
        return {}

    # (Optional) visualize polygons
    # visualize_polygons(polygons)

    print("\n[DEBUG] Step 5: Label each polygon with its lowest TIN surface...")
    labeled_polys = label_polygons_with_lowest_tin(polygons, decimated_tins, progress_callback)

    # Visualize labeled polygons
    # visualize_labeled_polygons(labeled_polys)

    print("\n[DEBUG] Step 6: Merge polygons by TIN label...")
    merged_by_tin = merge_polygons_by_tin(labeled_polys, progress_callback)

    print("\n[DEBUG] Final Visualization (TINs + intersections + legends)...")
    # visualize_tins_with_checkboxes(decimated_tins, intersections=snapped_intersections, global_scale=global_scale)

    print("\n[DEBUG] Step 7: Export results to DXF...")
    export_final_polygons(output_dxf, contours, snapped_intersections, merged_by_tin, progress_callback)

    print(f"\n[DEBUG] Workflow complete. Results saved to '{output_dxf}'.")
    if progress_callback:
        progress_callback(f"Workflow complete. Results saved to '{output_dxf}'.")

    return merged_by_tin

def filter_used_intersections(intersection_lines, merged_polygons, tolerance=0.1):
    """
    From the given list of intersection_lines (LineString objects in 3D),
    returns only those intersections that appear along the boundaries of the final (merged) polygons.
    
    This version uses a prepared geometry for fast spatial queries and prints progress.
    
    Parameters:
        intersection_lines (list): List of LineString objects (with 3D coordinates).
        merged_polygons (dict): Dictionary of merged polygons keyed by TIN label.
        tolerance (float): Tolerance distance to decide if a line is part of a boundary.
        
    Returns:
        used_intersections (list): List of LineString objects that were used.
    """
    # 1) Extract all polygon boundaries (in 2D) from merged polygons
    boundaries = []
    for geom in merged_polygons.values():
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            boundaries.append(geom.boundary)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                boundaries.append(poly.boundary)
    
    if not boundaries:
        print("[DEBUG] No polygon boundaries found!")
        return []
    
    # 2) Create a union of all boundaries and buffer it once
    final_boundaries = unary_union(boundaries)
    final_boundaries_buffered = final_boundaries.buffer(tolerance)
    prepared_boundaries = prep(final_boundaries_buffered)
    
    used_intersections = []
    total = len(intersection_lines)
    print(f"[DEBUG] Filtering used intersections: {total} intersections to check.")
    
    # 3) Check each intersection line (converted to 2D) using the prepared geometry
    for idx, line in enumerate(intersection_lines, start=1):
        # Convert the 3D line to 2D (drop the Z coordinate)
        line2d = LineString([(pt[0], pt[1]) for pt in line.coords])
        # If the buffered boundaries completely cover the intersection line, count it as used
        if prepared_boundaries.covers(line2d):
            used_intersections.append(line)
        
        # Print progress every 10%
        if idx % max(1, total // 10) == 0:
            print(f"[DEBUG] Processed {idx}/{total} intersections.")

    print(f"[DEBUG] Filtered {len(used_intersections)} used intersections out of {total}.")
    return used_intersections
