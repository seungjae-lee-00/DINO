import os
import pandas as pd
import xml.etree.ElementTree as elemTree
import torch, torchvision
import datasets.transforms as T
import datasets.strad_class_info

from PIL import Image
import json

# ann_path = '/media/hdd/leetop/data/GODTrain211111_filtered.csv'
# data_root = '/media/hdd/leetop/data/'

class StradDetection(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_root, transforms=None):
        super(StradDetection, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self._transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.img[idx]
        img_path = os.path.join(self.data_root, "JPEGImages", img_name)
        img = Image.open(img_path)
        ann_name = img_name.replace("jpg","xml")
        ann_path = os.path.join(self.data_root, "Annotations", ann_name)
        
        boxes, cls_ids = self.get_ann_info(ann_path)
        target = self._preprocess(boxes, cls_ids)
        w, h = img.size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # target["image_id"] = img_name.split('.')[0]

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def get_ann_info(self, ann_name):
        tree = elemTree.parse(ann_name)
        objects = tree.findall('object')

        boxes = []
        cls_ids = []

        for obj in objects:
            cls_name = obj.find('name').text
            
            if cls_name in datasets.strad_class_info.class_string:
                cls_idx_val = datasets.strad_class_info.class_string.index(cls_name)
                if datasets.strad_class_info.class_index[cls_idx_val] != -1:
                    cls_id = datasets.strad_class_info.class_index[cls_idx_val]
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymin = float(bbox.find('ymin').text)
                    ymax = float(bbox.find('ymax').text)
                    boxes.append([xmin, ymin, xmax, ymax])
                    cls_ids.append(cls_id)

        return boxes, cls_ids

    @staticmethod
    def _preprocess(boxes, cls_ids):
        bboxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        class_labels = torch.tensor(cls_ids, dtype=torch.int64)
        iscrowd = torch.zeros_like(class_labels)
        result = {"boxes" : bboxes,
               "labels" : class_labels,
               "iscrowd" : iscrowd
        }
        return result

def make_strad_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)

    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])
        
        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


