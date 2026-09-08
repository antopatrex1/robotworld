"""Camera-facing text cards, avoiding browser-dependent SDF font rendering."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def text_card(text):
    font_path=Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
    font=ImageFont.truetype(str(font_path),32) if font_path.exists() else ImageFont.load_default(size=32)
    bounds=font.getbbox(text)
    width=bounds[2]+28;height=54
    image=Image.new('RGBA',(width,height),(20,29,34,245))
    draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((0,0,width-1,height-1),radius=9,outline=(64,219,188,255),width=2)
    draw.text((14,8-bounds[1]),text,font=font,fill=(244,248,247,255))
    return np.asarray(image),width/height
