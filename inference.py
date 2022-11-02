import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
import datasets.strad_class_info

model_config_path = "logs/DINO/R50-MS4/config_cfg.py" # change the path of the model config file
model_checkpoint_path = "logs/DINO/R50-MS4/checkpoint.pth" # change the path of the model checkpoint
# model_config_path = "config/DINO/DINO_4scale_swin.py" # change the path of the model config file
# model_checkpoint_path = "model_zoo/checkpoint0029_4scale_swin.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.
args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
# args.backbone_dir = 'model_zoo/'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()
# load coco names
# with open('util/coco_id2name.json') as f:
#     id2name = json.load(f)
#     id2name = {int(k):v for k,v in id2name.items()}
id2name = datasets.strad_class_info.id2name
args.dataset_file = 'coco'
args.coco_path = "/data/leetop/data/coco" # the path of coco
args.fix_size = False

# dataset_val = build_dataset(image_set='val', args=args)

# image, targets = dataset_val[0]
# # build gt_dict for vis
# box_label = [id2name[int(item)] for item in targets['labels']]
# gt_dict = {
#     'boxes': targets['boxes'],
#     'image_id': targets['image_id'],
#     'size': targets['size'],
#     'box_label': box_label,
# }
# vslzr = COCOVisualizer()
# vslzr.visualize(image, gt_dict, savedir=None)

# output = model.cuda()(image[None].cuda())
# output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
# thershold = 0.5 # set a thershold

# scores = output['scores']
# labels = output['labels']
# boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
# select_mask = scores > thershold
# box_label = [id2name[int(item)] for item in labels[select_mask]]
# pred_dict = {
#     'boxes': boxes[select_mask],
#     'size': targets['size'],
#     'box_label': box_label
# }
# vslzr.visualize(image, pred_dict, savedir=None)

from PIL import Image
import datasets.transforms as T
import torchvision
import tqdm

# args.dataset_dir = os.path.join(args.coco_path,"val2017")
args.dataset_dir = "/data/udb/JPEGImages/"
# fnames = [filename for filename in os.listdir(args.dataset_dir) if os.path.isfile(os.path.join(args.dataset_dir, filename))]
import pandas as pd
df = pd.read_csv('GODTrain211111.csv')
file_list = df.sample(100).img.tolist()
fnames = [os.path.join(args.dataset_dir,fname) for fname in file_list]
for idx, fname in tqdm.tqdm(enumerate(fnames)):
    image = Image.open(os.path.join(args.dataset_dir, fname)).convert("RGB") # load image
    w, h = image.size
    if h > w :
        new_h = 1280
        new_w = int(1280*w/h)
    else :
        new_w = 1280
        new_h = int(1280*h/w)
    image = image.resize((new_w, new_h))

    # transform images
    transform = torchvision.transforms.Compose([
        # T.RandomResize([800], max_size=1333),
        # T.Resize(1280, max_size=1280),
        # torchvision.transforms.Resize([768], max_size=1280),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # predict images
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    # visualize outputs
    thershold = 0.3 # set a thershold

    vslzr = COCOVisualizer()

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    box_label = [id2name[int(item)] for item in labels[select_mask] if item != 0]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label,
        'image_id' : idx
    }
    vslzr.visualize(image, pred_dict, savedir="./infer_result", dpi=300)