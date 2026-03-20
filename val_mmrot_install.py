from mmdet.apis import init_detector, inference_detector
import mmrotate

config_file = './configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = './checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
inference_detector(model, 'demo/demo.jpg')

