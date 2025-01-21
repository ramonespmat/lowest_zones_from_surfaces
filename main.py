import os
import numpy as np
from scipy.spatial import Delaunay
import ezdxf
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union, linemerge, cascaded_union
import pyvista as pv

# Constants
INPUT_DXF = "contours_extra.dxf"
OUTPUT_DXF = "output/lowest_zones.dxf"
TIN_RESOLUTION = 10    # Not strictly used in this example, but left for context
INTERSECTION_LAYER = "INTERSECTIONS"
SNAP_TOLERANCE = 0.1   # e.g., 0.1 units for snapping

###############################################################################
# STEP 1: READ DXF CONTOURS
###############################################################################

def read_dxf_contours(file_path):
    """
    Reads contours from a DXF file and returns them grouped by layer.
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

        # Handle LWPOLYLINE
        if entity.dxftype() == "LWPOLYLINE":
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            if layer_name not in contours:
                contours[layer_name] = []
            elevation = entity.dxf.elevation
            coords = [(p[0], p[1], elevation) for p in entity.get_points()]
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found LWPOLYLINE in layer '{layer_name}' with elevation={elevation} and {len(coords)} pts.")

        # Handle 3D POLYLINE
        elif entity.dxftype() == "POLYLINE" and entity.is_3d_polyline:
            layer_name = entity.dxf.layer
            found_layers.add(layer_name)
            if layer_name not in contours:
                contours[layer_name] = []
            coords = [
                (
                    v.dxf.location.x,
                    v.dxf.location.y,
                    v.dxf.location.z
                )
                for v in entity.vertices
            ]
            contours[layer_name].append(coords)
            print(f"[DEBUG]   Found 3D POLYLINE in layer '{layer_name}' with {len(coords)} pts.")

        # Skip others
        else:
            pass

    print(f"\n[DEBUG] Total layers in DXF: {len(found_layers)} => {found_layers}")
    print(f"[DEBUG] Total entities in DXF: {found_entities}")
    print(f"[DEBUG] Total valid contour layers: {len(contours)}")
    return contours


###############################################################################
# STEP 2: GENERATE TIN FROM CONTOURS (for intersection)
###############################################################################

def generate_tin_from_contours(contours):
    """
    Creates TIN surfaces for each layer. We use XY for triangulation via scipy.spatial.Delaunay.
    Returns a dict: { layer_name: {"points": points, "triangles": simplices}, ... }
    """
    tins = {}
    for layer, layer_contours in contours.items():
        print(f"[DEBUG] Generating TIN for layer '{layer}'...")
        points = np.array([pt for contour_line in layer_contours for pt in contour_line])
        if len(points) < 3:
            print(f"[DEBUG]   Layer '{layer}' has fewer than 3 pts, skipping.")
            continue

        xy = points[:, :2]
        tri = Delaunay(xy)
        tins[layer] = {
            "points": points,
            "triangles": tri.simplices
        }
        print(f"[DEBUG]   TIN built with {len(tri.simplices)} triangles / {len(points)} pts in layer '{layer}'.")
    return tins


###############################################################################
# STEP 3: INTERSECTION LINES (using PyVista)
###############################################################################

def calculate_intersection_lines(tins):
    """
    Calculate 3D intersection lines between each pair of TIN surfaces.
    Return list of shapely LineStrings (3D).
    """
    print("[DEBUG] Calculating intersection lines between TIN surfaces...")
    layers = list(tins.keys())
    intersections = []

    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            layer1, layer2 = layers[i], layers[j]
            print(f"[DEBUG]   Checking intersection: '{layer1}' vs '{layer2}'")

            pts1 = tins[layer1]["points"]
            tri1 = tins[layer1]["triangles"]
            pts2 = tins[layer2]["points"]
            tri2 = tins[layer2]["triangles"]

            mesh1 = pv.PolyData(pts1)
            mesh1.faces = np.hstack([[3] + list(t) for t in tri1])
            mesh2 = pv.PolyData(pts2)
            mesh2.faces = np.hstack([[3] + list(t) for t in tri2])

            intersection_line, _, _ = mesh1.intersection(mesh2, split_first=False, split_second=False)

            if isinstance(intersection_line, pv.PolyData):
                if intersection_line.n_points > 1:
                    print(f"[DEBUG]     => Intersection: {intersection_line.n_points} pts, {intersection_line.n_cells} lines.")
                    for cidx in range(intersection_line.n_cells):
                        cell = intersection_line.extract_cells(cidx)
                        if cell.n_points > 1:
                            arr = np.array(cell.points)
                            intersections.append(LineString(arr))

    print(f"[DEBUG] Total intersection lines found: {len(intersections)}")
    return intersections


###############################################################################
# STEP 3A: SNAP INTERSECTION LINES TO NEAREST ORIGINAL CONTOUR
###############################################################################

from shapely.geometry import Point

def snap_intersections_to_contours(intersections, contours, tolerance=SNAP_TOLERANCE):
    """
    Snaps each vertex of the intersection lines to the nearest original contour
    line if within 'tolerance'. Returns new snapped intersection lines.
    """
    print("[DEBUG] Snapping intersection lines to original contours. Tolerance =", tolerance)
    # Build shapely lines from all original contours
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
                # Project onto best_line
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
# STEP 4: COMBINE ALL LINES AND POLYGONIZE
###############################################################################

from shapely.ops import linemerge, unary_union, polygonize

def build_planar_polygons(contours, snapped_intersections):
    """
    Combine original contour lines + intersection lines in 2D, polygonize them,
    and return a list of Shapely polygons.
    """
    print("[DEBUG] Building planar polygons from lines...")

    # 1) Gather all lines in 2D (ignore Z for polygonization)
    all_lines_2d = []
    # Original contour lines
    for layer, lines in contours.items():
        for coords in lines:
            coords_2d = [(x, y) for (x, y, z) in coords]
            all_lines_2d.append(LineString(coords_2d))
    # Snapped intersection lines
    for line in snapped_intersections:
        coords_2d = [(x, y) for (x, y, z) in line.coords]
        all_lines_2d.append(LineString(coords_2d))

    # 2) Merge into a single MultiLineString
    merged = unary_union(all_lines_2d)
    # 3) Polygonize
    polys = list(polygonize(merged))

    print(f"[DEBUG] => Found {len(polys)} polygons from polygonize().")
    return polys


###############################################################################
# STEP 5: FIND WHICH TIN IS LOWEST FOR EACH POLYGON
###############################################################################

def interpolate_tin_z(x, y, tin):
    """
    Simple method: find nearest triangle from 'tin' and do a barycentric interpolation
    to get approximate Z at (x, y). If outside hull or no triangle, return None.

    This is a DEMO approach. Production code might need robust point-in-triangle tests.
    """
    points = tin["points"]
    tri = tin["triangles"]

    # Quick nearest search: we loop over triangles to see if (x,y) is in that triangle's XY plane
    # For large data, a spatial index or advanced library is recommended.
    for simplex in tri:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]

        # We'll work in XY for the 2D test
        x0, y0 = p0[0], p0[1]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        # Barycentric approach: Check if (x,y) inside triangle [ (x0,y0), (x1,y1), (x2,y2) ]
        denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
        if abs(denom) < 1e-12:
            # Degenerate triangle?
            continue

        a = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denom
        b = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denom
        c = 1 - a - b

        # If 0 <= a,b,c <= 1 => inside
        if (a >= 0) and (b >= 0) and (c >= 0):
            # Interpolate Z
            z0, z1, z2 = p0[2], p1[2], p2[2]
            z = a*z0 + b*z1 + c*z2
            return z
    # Not found
    return None

def label_polygons_with_lowest_tin(polygons, tins):
    """
    For each polygon, compute its centroid (x,y). Interpolate Z from each TIN.
    Label polygon with the layer name having the minimal Z. Return a list of (polygon, layer).
    """
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
                if (best_z is None) or (z_val < best_z):
                    best_z = z_val
                    best_layer = layer

        if best_layer is not None:
            labeled.append((poly, best_layer))
        else:
            # If the polygon doesn't lie within any TIN's boundary
            labeled.append((poly, "NO_TIN"))

        print(f"  Polygon #{i} => lowest = {best_layer}")

    return labeled


###############################################################################
# STEP 6: MERGE ADJACENT POLYGONS WITH THE SAME TIN
###############################################################################

from shapely.ops import unary_union

def merge_polygons_by_tin(labeled_polys):
    """
    Merge polygons that share an edge and have the same TIN label.
    Returns a dict: { layer_name: MultiPolygon or Polygon } for each TIN.
    """
    print("[DEBUG] Merging adjacent polygons that share the same TIN label...")
    # Group by layer
    from collections import defaultdict
    grouping = defaultdict(list)
    for poly, layer in labeled_polys:
        grouping[layer].append(poly)

    # Merge within each layer
    merged_result = {}
    for layer, polys in grouping.items():
        # unary_union merges all polygons in that layer
        merged_poly = unary_union(polys)
        merged_result[layer] = merged_poly
        print(f"[DEBUG]   {len(polys)} polygons merged into one geometry for layer '{layer}'")

    return merged_result




###############################################################################
# EXPORT FINAL POLYGONS (OPTIONAL)
###############################################################################

def export_final_polygons(dxf_file, contours, intersection_lines, merged_dict):
    """
    Export:
      1) Original lines (as is)
      2) Intersection lines (on INTERSECTION_LAYER)
      3) Merged polygons for each TIN label
    """
    print("[DEBUG] Exporting final results to DXF:", dxf_file)
    doc = ezdxf.new()
    msp = doc.modelspace()

    # 1) Original lines on their layers
    for layer_name, lines in contours.items():
        for coords in lines:
            msp.add_polyline3d(coords, dxfattribs={"layer": layer_name})

    # 2) Intersection lines on INTERSECTION_LAYER
    for i, line in enumerate(intersection_lines, start=1):
        coords = list(line.coords)
        msp.add_polyline3d(coords, dxfattribs={"layer": INTERSECTION_LAYER})

    # 3) Merged polygons: we can flatten them to 2D polylines or export as 3D with Z=0
    # Each TIN label => new layer
    for layer_label, geom in merged_dict.items():
        # geom could be Polygon or MultiPolygon
        # We'll unify the logic by ensuring we iterate over polygons.
        if geom.is_empty:
            continue

        # shapely can have multiple polygons inside a MultiPolygon
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            polygons = []

        for poly in polygons:
            if poly.is_empty:
                continue
            # For each polygon, add a closed polyline in 2D (Z=0).
            xys = list(poly.exterior.coords)
            # convert them to 3D with Z=0
            coords_3d = [(x, y, 0.0) for (x, y) in xys]
            msp.add_polyline3d(coords_3d, dxfattribs={"layer": f"FINAL_{layer_label}"})

            # If the polygon has holes (interiors), optionally export them too
            for interior in poly.interiors:
                xys_int = list(interior.coords)
                coords_int_3d = [(x, y, 0.0) for (x, y) in xys_int]
                msp.add_polyline3d(coords_int_3d, dxfattribs={"layer": f"FINAL_{layer_label}_HOLE"})

    doc.saveas(dxf_file)
    print("[DEBUG] DXF export complete.")

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



###############################################################################
# MAIN WORKFLOW
###############################################################################

def main():
    # 1) Read DXF contours
    print("[DEBUG] Step 1: Reading contours...")
    contours = read_dxf_contours(INPUT_DXF)

    # 2) Generate TIN surfaces
    print("\n[DEBUG] Step 2: Generating TIN surfaces...")
    tins = generate_tin_from_contours(contours)

    # 3) Calculate intersection lines
    print("\n[DEBUG] Step 3: Calculating intersection lines...")
    intersections = calculate_intersection_lines(tins)

    # 3a) Snap intersection lines to original contours
    print("\n[DEBUG] Step 3a: Snapping intersection lines to original contours...")
    snapped_intersections = snap_intersections_to_contours(intersections, contours, tolerance=SNAP_TOLERANCE)

    # 4) Build planar polygons from all lines
    print("\n[DEBUG] Step 4: Building planar polygons from lines (original + intersection)...")
    polygons = build_planar_polygons(contours, snapped_intersections)

    # 5) Determine which TIN is lowest for each polygon
    print("\n[DEBUG] Step 5: Label each polygon with its lowest TIN surface...")
    labeled_polys = label_polygons_with_lowest_tin(polygons, tins)

    # 6) Merge adjacent polygons that share the same TIN label
    print("\n[DEBUG] Step 6: Merge polygons by TIN label...")
    merged_by_tin = merge_polygons_by_tin(labeled_polys)

    # 8) Visualize everything in 3D 
    visualize_tins(tins, snapped_intersections)

    # 7) Export results
    print("\n[DEBUG] Step 7: Export results to DXF...")
    os.makedirs(os.path.dirname(OUTPUT_DXF), exist_ok=True)
    export_final_polygons(OUTPUT_DXF, contours, snapped_intersections, merged_by_tin)

    print(f"\n[DEBUG] Workflow complete. Results saved to '{OUTPUT_DXF}'.")


if __name__ == "__main__":
    main()
