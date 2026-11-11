import json
import cv2
import pymupdf as fitz
from paddleocr import PaddleOCR
import paddle
import os
from paddleocr import TableCellsDetection
from paddleocr import LayoutDetection
import pymupdf as fitz


det_path = "./output/table_detection_res.json"
pdf_path = "./Incoming data/SBS_DETRITUS_202605.pdf"

model = LayoutDetection(
    model_name = "PP-DocLayout_plus-L",
    enable_mkldnn=False
    )

output = model.predict(pdf_path)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/table_detection_res.png")
    res.save_to_json(save_path=det_path)
    
with open (det_path, "r",encoding = "utf-8") as f:
    data = json.load(f)

pdf = fitz.open(pdf_path)
page = pdf[0]
rect = fitz.Rect(0,0,300,400)

out = page.get_pixmap(clip=rect,dpi=200)

for box in data["boxes"]:
    if box["label"] == "table":
        rect = fitz.Rect(box["coordinate"][0]/2,box["coordinate"][1]/2,box["coordinate"][2]/2,box["coordinate"][3]/2)

out = page.get_pixmap(clip=rect,dpi=200)
out.save("table.png")

model = TableCellsDetection(
    model_name="RT-DETR-L_wired_table_cell_det",
    enable_mkldnn=False
    )
output = model.predict("table.png", threshold=0.3, batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/cell_res.png")
    res.save_to_json("./output/cell_res.json")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

result = ocr.predict("table.png")
for res in result:
    res.print()
    res.save_to_img(save_path="./output/extracted_res.png")
    res.save_to_json(save_path="./output/extracted_res.json")
