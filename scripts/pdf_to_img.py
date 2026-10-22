import fitz
import os

PDF = r"F:\BS\docs_lm_ko\data\raw\test.pdf"
OUT = r"F:\BS\docs_lm_ko\outputs\pages"
os.makedirs(OUT, exist_ok=True)

doc = fitz.open(PDF)
for page in doc:
    pix = page.get_pixmap(dpi=120)
    path = os.path.join(OUT, f"page{page.number+1}.png")
    pix.save(path)
    print("saved:", path)
    page.set_rotation(0)