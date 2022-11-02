coco_path=$1
backbone_dir=$2
python -m torch.distributed.launch --nproc_per_node=8 main_sv.py \
	--output_dir logs/DINO/swin-MS4-strad -c config/DINO/DINO_strad.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir
