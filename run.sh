# CONFIG_FILE=/home/ubuntu/faster_rcnn/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
# python tools/train.py \
#     ${CONFIG_FILE} \

CONFIG_FILE=/home/ubuntu/faster_rcnn/configs/faster_rcnn/faster_rcnn_r50_fpn_b4x4_coco.py
GPU_NUM=1
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM}
