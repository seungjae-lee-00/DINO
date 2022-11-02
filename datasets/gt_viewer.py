import os
import json
from PIL import Image, ImageDraw



def draw_boxes(image, point1, point2, color=(0,0,255)):
    draw = ImageDraw.Draw(image)
    draw.rectangle((point1, point2), outline=color, width=2)
    return image

json_path = '/data/udb/GODTest_SVKPI_3000km/infer_result/recorder_2020-08-10_16-09-00_m1.avi_02100.json'
img_root = '/data/udb/GODTest_SVKPI_3000km/JPEGImages'
img_path = os.path.join(img_root, os.path.basename(json_path).replace('json','png'))

img = Image.open(img_path)

with open(json_path, 'r') as json_file:
    json_data = json.load(json_file)

for obj in json_data['objects'] :
    bbox = obj['2D box rectangle']
    img = draw_boxes(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]))

img.save('test.jpg')