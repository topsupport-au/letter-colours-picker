"""
extract_balloon_shape.py
Builds balloon_shape.json from "sphere_balloon.3mf" and writes it as BALLOON_SHAPE into
balloons.html and the Shopify widget (shopify/snippets/bubble-balloons-widget-content.liquid).

The balloon lies on its flat back on the build plate, so the same top-down
height map as extract_icon_shapes.py gives its wall-facing silhouette and a
baked lighting overlay. The mesh is first turned in the XY plane so the
balloon's long axis is vertical with the knot at the bottom, the way it hangs.
"""
import zipfile, json, os, re
import numpy as np
from extract_icon_shapes import icon_mesh, shape_entry, HERE

SRC = os.path.join(HERE, 'sphere_balloon.3mf')
OUT = os.path.join(HERE, 'balloon_shape.json')
PAGES = [os.path.join(HERE, 'balloons.html'),
         os.path.join(HERE, 'shopify', 'snippets', 'bubble-balloons-widget-content.liquid')]
OBJECT_ID = '16'     # one of the three balloon copies (11, 14, 16) on the plate


def hang_upright(verts):
    """Rotate about Z so the long axis is vertical and the narrow (knot) end is at the bottom."""
    xy = verts[:, :2] - verts[:, :2].mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(xy.T))
    axis = vecs[:, -1]                                   # long axis
    along = xy @ axis
    # The knot end is the one whose far 10% is thinnest across the axis
    across = xy @ np.array([-axis[1], axis[0]])
    lo, hi = np.quantile(along, [0.1, 0.9])
    if np.ptp(across[along >= hi]) < np.ptp(across[along <= lo]):
        axis = -axis                                     # make +axis point away from the knot
    ang = np.arctan2(axis[1], axis[0])
    rot = np.pi / 2 - ang                                # +axis -> +y (up)
    c, s = np.cos(rot), np.sin(rot)
    out = verts.copy()
    out[:, 0] = xy[:, 0] * c - xy[:, 1] * s
    out[:, 1] = xy[:, 0] * s + xy[:, 1] * c
    return out


if __name__ == '__main__':
    verts, tris = icon_mesh(zipfile.ZipFile(SRC), OBJECT_ID)
    verts = hang_upright(verts)
    entry = shape_entry(verts, tris)
    with open(OUT, 'w') as f:
        json.dump(entry, f, separators=(',', ':'))
    print(f"Balloon: {len(entry['paths'])} path(s)  viewBox 0 0 {entry['vw']} {entry['vh']}  "
          f"overlay {len(entry['img']['src']) // 1024} KB -> {OUT}")

    shape = json.dumps(entry, separators=(',', ':'))
    for page_path in PAGES:
        with open(page_path, encoding='utf-8') as f:
            page = f.read()
        page, n = re.subn(r'^([ \t]*)const BALLOON_SHAPE = .*;$',
                          lambda m: m.group(1) + 'const BALLOON_SHAPE = ' + shape + ';', page, count=1, flags=re.M)
        assert n == 1, 'BALLOON_SHAPE line not found in ' + page_path
        with open(page_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(page)
        print(f"Updated {page_path}")
