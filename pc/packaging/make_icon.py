"""Gera o ícone (ondas de rádio saindo de um ponto) em .ico e .png."""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, S - 1, S - 1], radius=48, fill=(22, 23, 27, 255))
c = S // 2
for r, col, w in ((48, (79, 195, 247), 14), (82, (30, 136, 229), 14)):
    d.arc([c - r, c - r, c + r, c + r], 135, 225, fill=col, width=w)
    d.arc([c - r, c - r, c + r, c + r], -45, 45, fill=col, width=w)
d.ellipse([c - 17, c - 17, c + 17, c + 17], fill=(255, 179, 0))
img.save(os.path.join(HERE, "sigid.png"))
img.save(os.path.join(HERE, "sigid.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("ícone gerado")
