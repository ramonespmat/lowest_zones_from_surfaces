import os
import numpy as np
from scipy.spatial import Delaunay
import ezdxf
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union, linemerge, cascaded_union
import pyvista as pv

# Constants
INPUT_DXF = "test2.dxf"
OUTPUT_DXF = "output/lowest_zones.dxf"
TIN_RESOLUTION = 10
INTERSECTION_LAYER = "INTERSECTIONS"
SNAP_TOLERANCE = 0.1
DECIMATION_REDUCTION = 0.0

###############################################################################
# STEP 1: READ DXF CONTOURS WITH ENFORCED CLOSURE
###############################################################################

def read_dxf_contours(file_path):
    """
    Reads contours from a DXF file, enforces closed polygons, and returns them grouped by layer.
    Recognizes both LWPOLYLINE (2D + elevation) and 3D POLYLINE entities.
    """
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
# STEP 2: GENERATE TIN FROM CONTOURS (for intersection)
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
        tri = Delaunay(xy)
        tins[layer] = {"points": points, "triangles": tri.simplices}
        print(f"[DEBUG]   TIN built with {len(tri.simplices)} triangles / {len(points)} pts in layer '{layer}'.")
    return tins

###############################################################################
# VISUALIZE INITIAL (UNMODIFIED) TINS
###############################################################################

def visualize_initial_tins(tins):
    print("[DEBUG] Visualizing initial TIN surfaces...")
    plotter = pv.Plotter()
    for layer, tin in tins.items():
        points = tin["points"]
        triangles = tin["triangles"]
        mesh = pv.PolyData(points)
        mesh.faces = np.hstack([[3] + list(tri) for tri in triangles])
        mesh["Elevation"] = points[:, 2]
        plotter.add_mesh(mesh, show_edges=True, scalars="Elevation", cmap="viridis", label=layer)
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
# STEP 3: INTERSECTION LINES (using *DECIMATED* TIN surfaces) (unchanged)
###############################################################################

def calculate_intersection_lines(decimated_tins):
    print("[DEBUG] Calculating intersection lines between decimated TIN surfaces...")
    layers = list(decimated_tins.keys())
    intersections = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            layer1, layer2 = layers[i], layers[j]
            print(f"[DEBUG]   Checking intersection: '{layer1}' vs '{layer2}'")
            pts1 = decimated_tins[layer1]["points"]
            tri1 = decimated_tins[layer1]["triangles"]
            pts2 = decimated_tins[layer2]["points"]
            tri2 = decimated_tins[layer2]["triangles"]
            mesh1 = pv.PolyData(pts1)
            mesh1.faces = np.hstack([[3] + list(t) for t in tri1])
            mesh2 = pv.PolyData(pts2)
            mesh2.faces = np.hstack([[3] + list(t) for t in tri2])
            intersection_line, _, _ = mesh1.intersection(mesh2, split_first=False, split_second=False)
            if isinstance(intersection_line, pv.PolyData) and intersection_line.n_points > 1:
                print(f"[DEBUG]     => Intersection: {intersection_line.n_points} pts, {intersection_line.n_cells} lines.")
                for cidx in range(intersection_line.n_cells):
                    cell = intersection_line.extract_cells(cidx)
                    if cell.n_points > 1:
                        arr = np.array(cell.points)
                        intersections.append(LineString(arr))
    print(f"[DEBUG] Total intersection lines found: {len(intersections)}")
    return intersections

###############################################################################
# STEP 3A: SNAP INTERSECTION LINES TO NEAREST ORIGINAL CONTOUR (unchanged)
###############################################################################

def snap_intersections_to_contours(intersections, contours, tolerance=SNAP_TOLERANCE):
    print("[DEBUG] Snapping intersection lines to original contours. Tolerance =", tolerance)
    original_lines = []
    for layer, lines in contours.items():
        for coords in lines:
            original_lines.append(LineString(coords))
    snapped = []
    total_snapped = 0
    for idx, line in enumerate(intersections, start=1):
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
        print(f"[DEBUG]   Intersection #{idx}: snapped {snap_count} points.")
    print(f"[DEBUG] Total snapped points across all intersection lines: {total_snapped}")
    return snapped

###############################################################################
# STEP 4: COMBINE ALL LINES AND POLYGONIZE with Detailed Debugging (unchanged)
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
# STEP 5: FIND WHICH TIN IS LOWEST FOR EACH POLYGON (unchanged)
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
# STEP 6: MERGE ADJACENT POLYGONS WITH THE SAME TIN (unchanged)
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
# EXPORT FINAL POLYGONS (unchanged)
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
# VISUALIZE TINS (unchanged)
###############################################################################

def visualize_tins(tins, intersections=None):
    print("[DEBUG] Visualizing final (decimated) TIN surfaces...")
    plotter = pv.Plotter()
    for layer, tin in tins.items():
        points = tin["points"]
        triangles = tin["triangles"]
        mesh = pv.PolyData(points)
        mesh.faces = np.hstack([[3] + list(tri) for tri in triangles])
        mesh["Elevation"] = points[:, 2]
        plotter.add_mesh(mesh, show_edges=True, scalars="Elevation", cmap="viridis", label=layer)
    if intersections:
        for line in intersections:
            coords = np.array(line.coords)
            if coords.shape[0] > 1:
                plotter.add_lines(coords, color="red", width=2)
    plotter.add_legend()
    plotter.show()

###############################################################################
# MAIN WORKFLOW
###############################################################################

def main():
    print("[DEBUG] Step 1: Reading contours...")
    contours = read_dxf_contours(INPUT_DXF)

    print("\n[DEBUG] Step 2: Generating TIN surfaces...")
    tins = generate_tin_from_contours(contours)

    visualize_initial_tins(tins)

    # For debugging, decimation is disabled
    decimated_tins = tins

    print("\n[DEBUG] Step 3: Calculating intersection lines...")
    intersections = calculate_intersection_lines(decimated_tins)

    print("\n[DEBUG] Step 3a: Snapping intersection lines to original contours...")
    snapped_intersections = snap_intersections_to_contours(intersections, contours, tolerance=SNAP_TOLERANCE)

    print("\n[DEBUG] Step 4: Building planar polygons from lines (original + intersection)...")
    polygons = build_planar_polygons(contours, snapped_intersections)

    visualize_polygons(polygons)

    print("\n[DEBUG] Step 5: Label each polygon with its lowest TIN surface...")
    labeled_polys = label_polygons_with_lowest_tin(polygons, decimated_tins)

    print("\n[DEBUG] Step 6: Merge polygons by TIN label...")
    merged_by_tin = merge_polygons_by_tin(labeled_polys)

    print("\n[DEBUG] Visualizing final TINs with intersections...")
    visualize_tins(decimated_tins, snapped_intersections)

    print("\n[DEBUG] Step 7: Export results to DXF...")
    os.makedirs(os.path.dirname(OUTPUT_DXF), exist_ok=True)
    export_final_polygons(OUTPUT_DXF, contours, snapped_intersections, merged_by_tin)

    print(f"\n[DEBUG] Workflow complete. Results saved to '{OUTPUT_DXF}'.")

if __name__ == "__main__":
    main()
