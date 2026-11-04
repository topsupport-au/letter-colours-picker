"""
extract_icon_shapes.py
Builds icon_shapes.json for icons.html from the icon 3mf files
("icons shapes.3mf", "icon ThePrincessSet.3mf", "icon Ocean.3mf", "icon Golf.3mf").

For each Iconic Bubble Icons product icon it z-buffers the top-down (XY)
height map of the printed mesh, then writes:
  - the silhouette as SVG paths (same normalised 0-100 units as letter_shapes.json)
  - a baked lighting overlay (WebP, grey + alpha) so the raised details --
    moon craters, astronaut visor, castle windows -- show in any colour.
The overlay is grey g at alpha a, chosen so that colour*(1-a) + g*a equals
colour*diffuse*(1-spec) + spec for every fill colour.
"""
import zipfile, re, json, os, io, base64, math
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'icon_shapes.json')

# Product option name (Iconic Bubble Icons "Icon" option) -> (3mf file, object id on its build plate).
# Several icons appear more than once in "icons shapes.3mf"; these are the copies on plate 1.
ICONS = [
    ('Butterfly',         'icons shapes.3mf',   '44'),
    ('Bow',               'icons shapes.3mf',   '40'),
    ('Crown',             'icons shapes.3mf',   '42'),
    ('Heart',             'icons shapes.3mf',   '38'),
    ('Flower',            'icons shapes.3mf',   '14'),
    ('Cloud',             'icons shapes.3mf',   '18'),
    ('Half Moon',         'icons shapes.3mf',   '53'),
    ('Shooting Star',     'icons shapes.3mf',   '45'),
    ('Full Moon',         'icons shapes.3mf',   '54'),
    ('Astronaut',         'icons shapes.3mf',   '51'),
    ('Rocket Detailed',   'icons shapes.3mf',   '50'),
    ('Rocket Simple',     'icons shapes.3mf',   '23'),
    ('Boat',              'icons shapes.3mf',   '34'),
    ('Car',               'icons shapes.3mf',   '26'),
    ('Dinosaur',          'icons shapes.3mf',   '20'),
    ('Princess Tiara',    'icon ThePrincessSet.3mf', '4'),
    ('Cinderella Shoe',   'icon ThePrincessSet.3mf', '10'),
    ('Princess Carriage', 'icon ThePrincessSet.3mf', '2'),
    ('Princess Castle',   'icon ThePrincessSet.3mf', '6'),
    ('Princess Wand',     'icon ThePrincessSet.3mf', '8'),
    ('Whale',             'icon Ocean.3mf',     '2'),
    ('Crab',              'icon Ocean.3mf',     '4'),
    ('Wave',              'icon Ocean.3mf',     '6'),
    ('Double Wave',       'icon Ocean.3mf',     '8'),
    ('Octopus',           'icon Ocean.3mf',     '10'),
    ('Submarine',         'icon Ocean.3mf',     '12'),
    ('Golf Ball',         'icon Golf.3mf',      '2'),
    ('Golf Cart',         'icon Golf.3mf',      '4'),
    ('Golf Club',         'icon Golf.3mf',      '6'),
    ('Golf Flag',         'icon Golf.3mf',      '8'),    # the single-piece flag, not the 3-piece print
    ('Golf Tee',          'icon Golf.3mf',      '16'),
]

GRID   = 240    # raster size of the longer side, pixels
MARGIN = 6      # pixel margin around the icon
DP_EPS = 0.3    # Douglas-Peucker tolerance (normalised units 0-100)
MIN_PX = 60     # minimum contour area in pixels to keep

# Light from the upper left, like the letter tool's specular highlight
LIGHT    = np.array([-0.45, 0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)
HALF     = LIGHT + np.array([0, 0, 1.0]); HALF /= np.linalg.norm(HALF)
AMBIENT  = 0.62
DIFFUSE  = 0.42
SPEC_K   = 0.85
SPEC_EXP = 45


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


def matrix(t):
    return np.array(t.split(), float).reshape(4, 3)


def icon_mesh(z, oid):
    """Vertices (build-plate coordinates, mm) and triangles of build item `oid`."""
    model = z.read('3D/3dmodel.model').decode()
    comps = re.search(r'<object id="' + oid + r'"[^>]*>\s*<components>(.*?)</components>', model, re.S).group(1)
    build = matrix(re.search(r'<item objectid="' + oid + r'"[^>]*transform="([^"]+)"', model).group(1))
    all_v, all_t, n = [], [], 0
    for path, sub, t in re.findall(r'p:path="([^"]+)" objectid="(\d+)"[^>]*transform="([^"]+)"', comps):
        xml = z.read(path.lstrip('/')).decode()
        s = xml.find('<object id="' + sub + '"'); e = xml.find('</object>', s)
        obj = xml[s:e]
        v = np.array(re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', obj), float)
        tr = np.array(re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', obj), int)
        C = matrix(t)
        v = v @ C[:3] + C[3]
        v = v @ build[:3] + build[3]
        all_v.append(v); all_t.append(tr + n); n += len(v)
    return np.vstack(all_v), np.vstack(all_t)


def height_map(verts, tris):
    """Top-down z-buffer. Returns (height mm, mask, mm per pixel, x0, y1)."""
    x0, x1 = verts[:, 0].min(), verts[:, 0].max()
    y0, y1 = verts[:, 1].min(), verts[:, 1].max()
    mm = max(x1 - x0, y1 - y0) / (GRID - 2 * MARGIN)
    W = int(math.ceil((x1 - x0) / mm)) + 2 * MARGIN + 1
    H = int(math.ceil((y1 - y0) / mm)) + 2 * MARGIN + 1
    px = (verts[:, 0] - x0) / mm + MARGIN
    py = (y1 - verts[:, 1]) / mm + MARGIN          # image rows go down
    pz = verts[:, 2]
    h = np.full((H, W), -np.inf)

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    edge = np.max(np.stack([np.hypot(px[a] - px[b], py[a] - py[b]),
                            np.hypot(px[b] - px[c], py[b] - py[c]),
                            np.hypot(px[c] - px[a], py[c] - py[a])]), axis=0)
    # Sample each triangle on a barycentric grid fine enough that no pixel inside it is skipped
    order = np.maximum(1, np.ceil(edge * 2)).astype(int)
    for k in np.unique(order):
        sel = order == k
        ta, tb, tc = a[sel], b[sel], c[sel]
        ij = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], float) / k
        wa, wb = ij[:, 0][None, :], ij[:, 1][None, :]
        wc = 1 - wa - wb
        sx = px[ta][:, None] * wa + px[tb][:, None] * wb + px[tc][:, None] * wc
        sy = py[ta][:, None] * wa + py[tb][:, None] * wb + py[tc][:, None] * wc
        sz = pz[ta][:, None] * wa + pz[tb][:, None] * wb + pz[tc][:, None] * wc
        np.maximum.at(h, (np.clip(sy.astype(int).ravel(), 0, H - 1),
                          np.clip(sx.astype(int).ravel(), 0, W - 1)), sz.ravel())
    mask = np.isfinite(h)
    mask = ndimage.binary_closing(mask, iterations=1) | mask   # pinholes between samples, not real holes
    floor = pz.min()
    h[~np.isfinite(h)] = floor
    # Pixels the closing added take their neighbours' height
    hole = mask & (h <= floor)
    if hole.any():
        h[hole] = ndimage.maximum_filter(h, size=3)[hole]
    return h, mask, mm, x0, y1


def overlay(h, mask, mm):
    """Grey+alpha lighting overlay as a WebP data URI."""
    hs = ndimage.gaussian_filter(h, 0.8)
    gy, gx = np.gradient(hs, mm)
    # Image rows run down while y runs up, so the y slope flips
    n = np.dstack([-gx, gy, np.ones_like(hs)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    diff = AMBIENT + DIFFUSE * np.clip(n @ LIGHT, 0, 1)
    spec = SPEC_K * np.clip(n @ HALF, 0, 1) ** SPEC_EXP
    # Darken toward the silhouette edge for roundness, as the letters' edge filter does
    edge = ndimage.gaussian_filter(mask.astype(float), 2.5)
    diff *= 0.82 + 0.18 * edge
    diff = np.clip(diff, 0, 1); spec = np.clip(spec, 0, 1)
    soft = np.clip(ndimage.gaussian_filter(mask.astype(float), 0.6) * 1.6 - 0.3, 0, 1)
    alpha = (1 - diff * (1 - spec)) * soft
    grey = np.where(alpha > 1e-3, spec * soft / np.maximum(alpha, 1e-3), 0)
    img = Image.fromarray(np.dstack([np.clip(grey * 255, 0, 255)] * 3 +
                                    [np.clip(alpha * 255, 0, 255)]).astype(np.uint8), 'RGBA')
    buf = io.BytesIO()
    img.save(buf, 'WEBP', quality=82, method=6)
    return 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode()


def silhouette(mask, norm_px):
    padded = np.pad(ndimage.gaussian_filter(mask.astype(float), 0.7), 1)
    paths = []
    for contour in measure.find_contours(padded, 0.5):
        if len(contour) < 6:
            continue
        rows, cols = contour[:, 0] - 1, contour[:, 1] - 1
        if abs(np.sum(cols[:-1]*rows[1:] - cols[1:]*rows[:-1])) / 2 < MIN_PX:
            continue
        pts = [(round((c - MARGIN) * norm_px, 2), round((r - MARGIN) * norm_px, 2)) for r, c in contour]
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        pts = douglas_peucker(pts, DP_EPS)
        if len(pts) >= 3:
            paths.append('M' + 'L'.join(f'{x},{y}' for x, y in pts) + 'Z')
    return paths


result = {}
zips = {}
for name, file, oid in ICONS:
    z = zips.setdefault(file, zipfile.ZipFile(os.path.join(HERE, file)))
    verts, tris = icon_mesh(z, oid)
    h, mask, mm, x0, y1 = height_map(verts, tris)
    w_mm = verts[:, 0].max() - verts[:, 0].min()
    h_mm = verts[:, 1].max() - verts[:, 1].min()
    norm_px = 100.0 / (max(w_mm, h_mm) / mm)       # pixels -> normalised units
    H, W = mask.shape
    result[name] = {
        'vw': round(w_mm / mm * norm_px, 2),
        'vh': round(h_mm / mm * norm_px, 2),
        'paths': silhouette(mask, norm_px),
        'img': {'x': round(-MARGIN * norm_px, 2), 'y': round(-MARGIN * norm_px, 2),
                'w': round(W * norm_px, 2), 'h': round(H * norm_px, 2), 'src': overlay(h, mask, mm)},
    }
    r = result[name]
    print(f"  {name}: {len(r['paths'])} path(s)  viewBox 0 0 {r['vw']} {r['vh']}  overlay {len(r['img']['src']) // 1024} KB", flush=True)

with open(OUT, 'w') as f:
    json.dump(result, f, separators=(',', ':'))
print(f"\nDone -- {OUT}  ({os.path.getsize(OUT) // 1024} KB, {len(result)} icons)")
