"""
extract_letter_svg.py
Rasterises the top-down projection of every bubble letter mesh,
traces the silhouette with marching-squares, and writes letter_shapes.json.
"""
import zipfile, re, json, os, math
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

LETTER_MAP = {
    'A':'2','B':'23','C':'21','D':'22','E':'25','F':'26',
    'G':'24','H':'6','I':'19','J':'20','K':'7','L':'17',
    'M':'4','N':'1','O':'10','P':'15','Q':'3','R':'11',
    'S':'14','T':'18','U':'8','V':'12','W':'5','X':'9',
    'Y':'16','Z':'13'
}
STL_PATHS = {
    '2':'3D/Objects/object_1011.model','23':'3D/Objects/object_167.model',
    '21':'3D/Objects/object_949.model','22':'3D/Objects/object_1007.model',
    '25':'3D/Objects/object_1006.model','26':'3D/Objects/object_594.model',
    '24':'3D/Objects/object_908.model','6':'3D/Objects/object_1017.model',
    '19':'3D/Objects/object_775.model','20':'3D/Objects/object_843.model',
    '7':'3D/Objects/object_940.model','17':'3D/Objects/object_1014.model',
    '4':'3D/Objects/object_1000.model','1':'3D/Objects/object_1016.model',
    '10':'3D/Objects/object_946.model','15':'3D/Objects/object_922.model',
    '3':'3D/Objects/object_4.model','11':'3D/Objects/object_948.model',
    '14':'3D/Objects/object_1002.model','18':'3D/Objects/object_782.model',
    '8':'3D/Objects/object_912.model','12':'3D/Objects/object_1013.model',
    '5':'3D/Objects/object_6.model','9':'3D/Objects/object_911.model',
    '16':'3D/Objects/object_788.model','13':'3D/Objects/object_761.model'
}

HERE = os.path.dirname(os.path.abspath(__file__))
TMF  = os.path.join(HERE, 'alphabet_MAIN_printing.3mf')
OUT  = os.path.join(HERE, 'letter_shapes.json')

GRID   = 512    # raster resolution
MARGIN = 8      # pixel margin around the letter
DP_EPS = 0.35   # Douglas-Peucker tolerance (normalised units 0-100)
MIN_PX = 200    # minimum contour area in pixels to keep (filters tiny holes/artifacts)


def douglas_peucker(pts, eps):
    if len(pts) <= 2:
        return pts
    p0, pn = pts[0], pts[-1]
    dx = pn[0] - p0[0];  dy = pn[1] - p0[1]
    denom = math.hypot(dx, dy) + 1e-12
    max_d, max_i = 0.0, 0
    for i in range(1, len(pts) - 1):
        p = pts[i]
        d = abs(dy*p[0] - dx*p[1] + pn[0]*p0[1] - pn[1]*p0[0]) / denom
        if d > max_d:
            max_d, max_i = d, i
    if max_d > eps:
        left  = douglas_peucker(pts[:max_i + 1], eps)
        right = douglas_peucker(pts[max_i:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def poly_area(pts):
    n = len(pts)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(a) / 2.0


def rasterise_letter(verts, tris_raw):
    """
    Project ALL triangles onto the XY plane and rasterise into a binary grid.
    Returns the PIL image and the (x0, y0, x1, y1) mesh bounding box.
    """
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0

    inner = GRID - 2 * MARGIN
    scale = inner / max(w, h)

    def to_px(x, y):
        px = int((x - x0) * scale + MARGIN)
        # Flip Y: SVG y increases downward, mesh y increases upward
        py = int((y1 - y) * scale + MARGIN)
        return (px, py)

    img = Image.new('L', (GRID, GRID), 0)
    draw = ImageDraw.Draw(img)

    for i1, i2, i3 in tris_raw:
        p1 = to_px(verts[i1][0], verts[i1][1])
        p2 = to_px(verts[i2][0], verts[i2][1])
        p3 = to_px(verts[i3][0], verts[i3][1])
        draw.polygon([p1, p2, p3], fill=255)

    return img, (x0, y0, x1, y1, scale)


def extract_contours(img, bounds):
    """
    Use marching squares to find contours in the raster image.
    Returns list of [(x_norm, y_norm)] closed paths in 0-100 normalised space.
    """
    x0, y0, x1, y1, scale = bounds
    arr = np.array(img, dtype=float) / 255.0

    # find_contours at level 0.5 gives boundaries between filled/unfilled
    contours = measure.find_contours(arr, 0.5)

    paths = []
    for contour in contours:
        # contour is (N,2) array of (row, col) = (y_px, x_px)
        # filter tiny contours
        if len(contour) < 6:
            continue
        # pixel area (shoelace on pixel coords)
        rows = contour[:, 0]
        cols = contour[:, 1]
        px_area = abs(np.sum(cols[:-1]*rows[1:] - cols[1:]*rows[:-1])) / 2
        if px_area < MIN_PX:
            continue
        # Convert pixel coords back to normalised 0-100 space
        pts = []
        for row, col in contour:
            # col = x_px = (x_mesh - x0)*scale + MARGIN  =>  x_mesh = (col-MARGIN)/scale + x0
            # row = y_px = (y1 - y_mesh)*scale + MARGIN  =>  y_mesh = y1 - (row-MARGIN)/scale
            x_mesh = (col - MARGIN) / scale + x0
            y_mesh = y1 - (row - MARGIN) / scale
            mesh_w = x1 - x0
            mesh_h = y1 - y0
            norm_scale = 100.0 / max(mesh_w, mesh_h)
            x_norm = round((x_mesh - x0) * norm_scale, 2)
            y_norm = round((y1 - y_mesh) * norm_scale, 2)  # y already flipped above
            pts.append((x_norm, y_norm))
        paths.append(pts)

    return paths


def build_svg_paths(paths_norm, eps=DP_EPS):
    """Simplify and build SVG 'd' strings from normalised path coords."""
    result = []
    for pts in paths_norm:
        # find_contours closes the loop (last pt == first pt); remove duplicate
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        simplified = douglas_peucker(pts, eps)
        if len(simplified) < 3:
            continue
        parts = [f"M{simplified[0][0]},{simplified[0][1]}"]
        for pt in simplified[1:]:
            parts.append(f"L{pt[0]},{pt[1]}")
        parts.append("Z")
        result.append("".join(parts))
    return result


def get_viewbox(paths_norm):
    all_pts = [p for path in paths_norm for p in path]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    # Already normalised to 0-100 scale
    return round(w, 2), round(h, 2)


print(f"Reading {TMF} ...")
result = {}

with zipfile.ZipFile(TMF) as z:
    for letter in sorted(LETTER_MAP):
        mid  = LETTER_MAP[letter]
        path = STL_PATHS[mid]
        xml  = z.read(path).decode()

        verts = [
            (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            for m in re.finditer(
                r'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"', xml)
        ]
        tris_raw = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(
                r'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"', xml)
        ]

        img, bounds = rasterise_letter(verts, tris_raw)
        paths_norm  = extract_contours(img, bounds)

        if not paths_norm:
            print(f"  {letter}: no contours found!")
            continue

        # Determine viewBox from actual extent of normalised paths
        vw, vh = get_viewbox(paths_norm)
        svg_paths = build_svg_paths(paths_norm)

        total_pts = sum(p.count('L') + 1 for p in svg_paths)
        print(f"  {letter}: {len(svg_paths)} path(s), {total_pts} pts  viewBox 0 0 {vw} {vh}")
        result[letter] = {'vw': vw, 'vh': vh, 'paths': svg_paths}

with open(OUT, 'w') as f:
    json.dump(result, f, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"\nDone — {OUT}  ({kb} KB)")
