import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
import datasets.strad_class_info

def scale_im(scale_size, im_size, multiple):
    im_size_min = min(im_size)
    im_size_max = max(im_size)
    scale_min = min(scale_size)
    scale_max = max(scale_size)
    
    im_scale = scale_min/im_size_min
    
    if round(im_scale*im_size_max>scale_max):
        im_scale = scale_max/im_size_max

    im_scale_x = int(im_scale*im_size[1]/multiple)*multiple
    im_scale_y = int(im_scale*im_size[0]/multiple)*multiple

    return im_scale_x, im_scale_y

    
model_config_path = "logs/DINO/R50-MS4/config_cfg.py" # change the path of the model config file
model_checkpoint_path = "logs/DINO/R50-MS4/checkpoint.pth" # change the path of the model checkpoint
args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()
id2name = datasets.strad_class_info.id2name
name2class_id = datasets.strad_class_info._class_id_map
args.dataset_file = 'coco'
args.coco_path = "/data/leetop/data/coco" # the path of coco
args.fix_size = False

from PIL import Image
import datasets.transforms as T
import torchvision
import tqdm

args.dataset_dir = "/data/udb/GODTest_SVKPI_3000km/JPEGImages/"
import pandas as pd

df = pd.read_csv('data_csv/svkpi3000km.csv')
# df = pd.read_csv('data_csv/svkpi3000km_mini.csv')
file_list = df['img'].to_list()
fnames = [os.path.join(args.dataset_dir,fname) for fname in file_list]
vslzr = COCOVisualizer()

for idx, fname in tqdm.tqdm(enumerate(fnames)):
    image = Image.open(os.path.join(args.dataset_dir, fname)).convert("RGB") # load image
    origin_w, origin_h = image.size
    new_w, new_h = scale_im([1376, 768], image.size, multiple=32)
    image = image.resize((new_h, new_w))
    
    # transform images
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image)
    # predict images
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    # visualize outputs
    thershold = 0.01 # set a thershold

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    scores = scores[select_mask]
    labels = labels[select_mask]
    bboxes = boxes[select_mask]
    
    objects = []
    for i, box in enumerate(bboxes):
        if labels[i] != 0:
            xyxy = [box[0].item()*origin_w - (box[2].item()*origin_w)/2, 
                    box[1].item()*origin_h - (box[3].item()*origin_h)/2,
                    box[0].item()*origin_w + (box[2].item()*origin_w)/2, 
                    box[1].item()*origin_h + (box[3].item()*origin_h)/2]

            object = {"2D box rectangle" : xyxy, 
                      "class_type" : name2class_id[id2name[int(labels[i])]],
                      "class_id" : name2class_id[id2name[int(labels[i])]],
                      "score" : scores[i].item()
            }
            objects.append(object)
    
    pred_dict = {
        "num_obj" : len(objects),
        "objects" : objects
    }

    out_name = os.path.basename(fname).replace('.png','.json')
    out_path = os.path.join("/data/udb/GODTest_SVKPI_3000km/infer_result/", out_name)

    with open(out_path, 'w') as json_file:
        json.dump(pred_dict, json_file)

    # box_label = [id2name[int(item)] for item in labels]
    # pred_dict = {
    #     'boxes': boxes[select_mask],
    #     'size': torch.Tensor([image.shape[1], image.shape[2]]),
    #     'box_label': box_label,
    #     'image_id' : idx
    # }
    # vslzr.visualize(image, pred_dict, savedir="./infer_result_svkpi_3000km", dpi=300)
