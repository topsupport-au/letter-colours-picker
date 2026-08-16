"""
extract_lowercase_svg.py
Rasterises the top-down projection of every lowercase bubble-letter mesh
in LowerCase_BL_6cm.3mf, traces the silhouette with marching-squares,
and writes lowercase_letter_shapes.json.

Object-file mapping was determined by rendering every distinct mesh in the
3mf to a labelled contact sheet and visually matching each shape to its
letter (the 3mf's internal object names don't identify letters directly).
A handful of extra meshes in the file (leftover boolean-cut artifacts /
reject drafts) aren't real letters and are excluded.
"""
import zipfile, re, json, os, math
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
TMF  = os.path.join(HERE, 'LowerCase_BL_6cm.3mf')
OUT  = os.path.join(HERE, 'lowercase_letter_shapes.json')

# letter -> (component .model file, objectid inside that file)
LETTER_MAP = {
    'a': ('object_122.model', '5'),
    'b': ('object_67.model',  '23'),
    'c': ('object_72.model',  '31'),
    'd': ('object_76.model',  '39'),
    'e': ('object_123.model', '15'),
    'f': ('object_83.model',  '52'),
    'g': ('object_86.model',  '57'),
    'h': ('object_62.model',  '7'),
    'i': ('object_126.model', '19'),
    'j': ('object_91.model',  '63'),
    'k': ('object_75.model',  '37'),
    'l': ('object_65.model',  '17'),
    'm': ('object_120.model', '3'),
    'n': ('object_87.model',  '59'),
    'o': ('object_61.model',  '1'),
    'p': ('object_64.model',  '13'),
    'q': ('object_69.model',  '27'),
    'r': ('object_74.model',  '35'),
    's': ('object_78.model',  '43'),
    't': ('object_82.model',  '50'),
    'u': ('object_124.model', '9'),
    'v': ('object_68.model',  '25'),
    'w': ('object_73.model',  '33'),
    'x': ('object_77.model',  '41'),
    'y': ('object_81.model',  '48'),
    'z': ('object_84.model',  '54'),
}

GRID   = 512
MARGIN = 8
DP_EPS = 0.35
MIN_PX = 200


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


def rasterise_letter(verts, tris_raw):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0

    inner = GRID - 2 * MARGIN
    scale = inner / max(w, h)

    def to_px(x, y):
        px = int((x - x0) * scale + MARGIN)
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
    x0, y0, x1, y1, scale = bounds
    arr = np.array(img, dtype=float) / 255.0
    contours = measure.find_contours(arr, 0.5)

    paths = []
    for contour in contours:
        if len(contour) < 6:
            continue
        rows = contour[:, 0]
        cols = contour[:, 1]
        px_area = abs(np.sum(cols[:-1]*rows[1:] - cols[1:]*rows[:-1])) / 2
        if px_area < MIN_PX:
            continue
        pts = []
        for row, col in contour:
            x_mesh = (col - MARGIN) / scale + x0
            y_mesh = y1 - (row - MARGIN) / scale
            mesh_w = x1 - x0
            mesh_h = y1 - y0
            norm_scale = 100.0 / max(mesh_w, mesh_h)
            x_norm = round((x_mesh - x0) * norm_scale, 2)
            y_norm = round((y1 - y_mesh) * norm_scale, 2)
            pts.append((x_norm, y_norm))
        paths.append(pts)

    return paths


def build_svg_paths(paths_norm, eps=DP_EPS):
    result = []
    for pts in paths_norm:
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
    return round(w, 2), round(h, 2)


print(f"Reading {TMF} ...")
result = {}

with zipfile.ZipFile(TMF) as z:
    for letter in sorted(LETTER_MAP):
        fname, oid = LETTER_MAP[letter]
        path = f'3D/Objects/{fname}'
        xml = z.read(path).decode()

        m = re.search(r'<object id="%s"[^>]*>(.*?)</object>' % re.escape(oid), xml, re.S)
        block = m.group(1) if m else xml

        verts = [
            (float(mm.group(1)), float(mm.group(2)), float(mm.group(3)))
            for mm in re.finditer(
                r'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"', block)
        ]
        tris_raw = [
            (int(mm.group(1)), int(mm.group(2)), int(mm.group(3)))
            for mm in re.finditer(
                r'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"', block)
        ]

        img, bounds = rasterise_letter(verts, tris_raw)
        paths_norm  = extract_contours(img, bounds)

        if not paths_norm:
            print(f"  {letter}: no contours found!")
            continue

        vw, vh = get_viewbox(paths_norm)
        svg_paths = build_svg_paths(paths_norm)

        total_pts = sum(p.count('L') + 1 for p in svg_paths)
        print(f"  {letter}: {len(svg_paths)} path(s), {total_pts} pts  viewBox 0 0 {vw} {vh}")
        result[letter] = {'vw': vw, 'vh': vh, 'paths': svg_paths}

with open(OUT, 'w') as f:
    json.dump(result, f, separators=(',', ':'))

kb = os.path.getsize(OUT) // 1024
print(f"\nDone — {OUT}  ({kb} KB)  {len(result)}/26 letters")
