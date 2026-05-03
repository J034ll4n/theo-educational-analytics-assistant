"""Regenera `assets/app.ico` a partir de `assets/app_icon.png` (recorte quadrado central).

Requer: pip install Pillow
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
ASSETS = _ROOT / "assets"


def main() -> None:
    png = ASSETS / "app_icon.png"
    if not png.exists():
        raise SystemExit(f"Falta {png}")
    img = Image.open(png).convert("RGBA")
    w, h = img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    img = img.crop((l, t, l + s, t + s))
    im256 = img.resize((256, 256), Image.Resampling.LANCZOS)
    out = ASSETS / "app.ico"
    im256.save(out, format="ICO", sizes=[(256, 256)])
    print(f"Escrito: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
