"""Generate the small synthetic fixtures used by the frontend end-to-end test."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests import synthetic as S

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(out, exist_ok=True)
a = S.scene((300, 260), 7)
b = S.add_patch(a, 180, 40, 260, 120)
open(os.path.join(out, "a.png"), "wb").write(S.png_bytes(a))
open(os.path.join(out, "b.png"), "wb").write(S.png_bytes(b))
open(os.path.join(out, "g.tif"), "wb").write(S.make_geotiff(a[:200, :200]))
print("fixtures written to", out)
