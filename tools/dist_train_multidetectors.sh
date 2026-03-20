#!/usr/bin/env bash

workdir_glidingvertex="../UAV-ROD_training/detection_model/detectors_640x360/gliding_vertex"
workdir_roitrans="../UAV-ROD_training/detection_model/detectors_640x360/roi_trans"
workdir_rotatedfasterrcnn="../UAV-ROD_training/detection_model/detectors_640x360/faster_rcnn_o"
workdir_rotatedfcos="../UAV-ROD_training/detection_model/detectors_640x360/fcos_o"
workdir_s2anet="../UAV-ROD_training/detection_model/detectors_640x360/s2anet"

config_glidingvertex="../UAV-ROD_training/configs/gliding_vertex/gliding_vertex_r50_fpn_1x_uavrod_le90.py"
config_roitrans="../UAV-ROD_training/configs/roi_trans/roi_trans_r50_fpn_1x_uavrod_le90.py"
config_rotatedfasterrcnn="../UAV-ROD_training/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_uavrod_le90.py"
config_rotatedfcos="../UAV-ROD_training/configs/rotated_fcos/rotated_fcos_r50_fpn_1x_uavrod_le90.py"
config_s2anet="../UAV-ROD_training/configs/s2anet/s2anet_r50_fpn_1x_uavrod_le90.py"

gpu0=0
gpu1=1
gpu_number=2

CUDA_VISIBLE_DEVICES=$gpu0,$gpu1 ./dist_train.sh $config_glidingvertex $gpu_number --work-dir $workdir_glidingvertex
CUDA_VISIBLE_DEVICES=$gpu0,$gpu1 ./dist_train.sh $config_roitrans $gpu_number --work-dir $workdir_roitrans
CUDA_VISIBLE_DEVICES=$gpu0,$gpu1 ./dist_train.sh $config_rotatedfasterrcnn $gpu_number --work-dir $workdir_rotatedfasterrcnn
CUDA_VISIBLE_DEVICES=$gpu0,$gpu1 ./dist_train.sh $config_rotatedfcos $gpu_number --work-dir $workdir_rotatedfcos
CUDA_VISIBLE_DEVICES=$gpu0,$gpu1 ./dist_train.sh $config_s2anet $gpu_number --work-dir $workdir_s2anet
