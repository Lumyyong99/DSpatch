import torch
from mmdet.apis import inference_detector
from ipdb import set_trace as st



def inference_det_loss_on_masked_images(model, adv_images_batch_t, img_metas, patch_labels_batch_t, patch_boxes_batch_t, model_name='retinanet_o'):
    """
    Inference image(s) with the detector.
    Args:
        model (MMRotate model): victim model load from mmrotate
        model_name (str)): name of the model, determine loss calculation
        adv_images_batch_t (Tensor): scene images batch for loss calculation, torch.Size([batch_size, 3, H, W]), should be RGB format
        img_metas (List): need to put in forward_train.
        patch_labels_batch_t (Tensor): Patch labels tensor. torch.Size([batch_size, patch_per_image, 1])
        patch_boxes_batch_t (Tensor): Patch bounding-boxes tensor. torch.Size([batch_size, patch_per_image, 5]) here 5 means [cx, cy, w, h, theta]. theta为正表示顺时针旋转
    Returns:
        loss (Tensor). Calculation loss of victim detector
    """
    adv_images_batch_t = adv_images_batch_t.cuda()
    patch_labels_batch_list = []
    patch_boxes_batch_list = []
    for i in range(patch_labels_batch_t.size(0)):
        patch_labels = patch_labels_batch_t[i, :, :].squeeze(-1).cuda()  # 将torch.Size([4, 16, 1])转换成torch.Size([4, 16])
        patch_boxes = patch_boxes_batch_t[i, :, :].cuda()
        patch_labels_batch_list.append(patch_labels.long())  # len = batch_size, 每个元素torch.Size([16])
        patch_boxes_batch_list.append(patch_boxes.float())  # len = batch_size, 每个元素torch.Size([16, 5])
    # forward the model
    with torch.set_grad_enabled(True):
        loss = model.forward_train(img=adv_images_batch_t, gt_bboxes=patch_boxes_batch_list, img_metas=img_metas, gt_labels=patch_labels_batch_list)

    # 求和
    if model_name == 'retinanet_o' or model_name == 'retinanet_swint':
        loss_cls_stack = torch.stack(loss['loss_cls'])
        loss_cls = torch.sum(loss_cls_stack)
        loss_bbox_stack = torch.stack(loss['loss_bbox'])
        loss_bbox = torch.sum(loss_bbox_stack)
        loss = 1.0 * loss_cls + 1.0 * loss_bbox
    elif model_name == 'faster_rcnn_o' or model_name == 'faster_rcnn_swint':
        loss_rpn_cls_stack = torch.stack(loss['loss_rpn_cls'])
        loss_rpn_cls = torch.sum(loss_rpn_cls_stack)
        loss_rpn_bbox_stack = torch.stack(loss['loss_rpn_bbox'])
        loss_rpn_bbox = torch.sum(loss_rpn_bbox_stack)
        loss_cls = loss['loss_cls']
        loss_bbox = loss['loss_bbox']
        loss = 1.0 * loss_cls + 1.0 * loss_bbox + 1.0 * loss_rpn_cls + 1.0 * loss_rpn_bbox
    # elif model_name == 'fcos_o':
    #     loss_cls_total = loss['loss_cls']
    #     loss_bbox_total = loss['loss_bbox']
    #     loss_centerness = loss['loss_centerness']
    #     loss = 1.0 * loss_cls_total + 1.0 * loss_bbox_total + 0.1 * loss_centerness
    # TODO: 修改这里
    elif model_name == 'gliding_vertex' or model_name == 'gliding_vertex_swint':
        loss_rpn_cls_stack = torch.stack(loss['loss_rpn_cls'])
        loss_rpn_cls = torch.sum(loss_rpn_cls_stack)
        loss_rpn_bbox_stack = torch.stack(loss['loss_rpn_bbox'])
        loss_rpn_bbox = torch.sum(loss_rpn_bbox_stack)
        loss_cls_total = loss['loss_cls']
        loss_bbox_total = loss['loss_bbox']
        loss_fix = loss['loss_fix']
        loss_ratio = loss['loss_ratio']
        loss = 1.0 * loss_cls_total + 1.0 * loss_bbox_total + 1.0 * loss_rpn_cls + 1.0 * loss_rpn_bbox + 1.0 * loss_fix + 1.0 * loss_ratio
    elif model_name == 'oriented_rcnn' or model_name == 'oriented_rcnn_swint':
        loss_rpn_cls_stack = torch.stack(loss['loss_rpn_cls'])
        loss_rpn_cls = torch.sum(loss_rpn_cls_stack)
        loss_rpn_bbox_stack = torch.stack(loss['loss_rpn_bbox'])
        loss_rpn_bbox = torch.sum(loss_rpn_bbox_stack)
        loss_cls = loss['loss_cls']
        loss_bbox = loss['loss_bbox']
        loss = 1.0 * loss_rpn_cls + 1.0 * loss_rpn_bbox + 1.0 * loss_cls + 1.0 * loss_bbox
    elif model_name == 'roi_trans' or model_name == 'roi_trans_swint':  # 两阶段
        loss_rpn_cls_stack = torch.stack(loss['loss_rpn_cls'])
        loss_rpn_cls = torch.sum(loss_rpn_cls_stack)
        loss_rpn_bbox_stack = torch.stack(loss['loss_rpn_bbox'])
        loss_rpn_bbox = torch.sum(loss_rpn_bbox_stack)
        s0_loss_cls = loss['s0.loss_cls']
        s0_loss_bbox = loss['s0.loss_bbox']
        s1_loss_cls = loss['s1.loss_cls']
        s1_loss_bbox = loss['s1.loss_bbox']
        loss = 1.0 * loss_rpn_cls + 1.0 * loss_rpn_bbox + 1.0 * s0_loss_cls + 1.0 * s0_loss_bbox + 1.0 * s1_loss_cls + 1.0 * s1_loss_bbox
    elif model_name == 's2anet' or model_name == 's2anet_swint':  # fam 和 odm，都是单阶段形式
        fam_loss_cls_stack = torch.stack(loss['fam.loss_cls'])
        fam_loss_cls = torch.sum(fam_loss_cls_stack)
        fam_loss_bbox_stack = torch.stack(loss['fam.loss_bbox'])
        fam_loss_bbox = torch.sum(fam_loss_bbox_stack)
        odm_loss_cls_stack = torch.stack(loss['odm.loss_cls'])
        odm_loss_cls = torch.sum(odm_loss_cls_stack)
        odm_loss_bbox_stack = torch.stack(loss['odm.loss_bbox'])
        odm_loss_bbox = torch.sum(odm_loss_bbox_stack)
        loss = 1.0 * fam_loss_cls + 1.0 * fam_loss_bbox + 1.0 * odm_loss_cls + 1.0 * odm_loss_bbox

    # requires grad
    loss.requires_grad_(True)
    return loss

def inference_2(model, adv_images_batch_t, img_metas, vanish_labels_list, vanish_bboxes_list):
    """
    Inference image(s) with the detector. used for vanish attack.
    Args:
        model (MMRotate model): victim model load from mmrotate
        adv_images_batch_t (Tensor): scene images batch for loss calculation, torch.Size([batch_size, 3, H, W]), should be RGB format
        img_metas (List): need to put in forward_train.
        vanish_labels_list (list): Patch labels list. list[empty, empty,...]
        vanish_bboxes_list (list): Patch bounding-boxes list. list[empty, empty,...]
    Returns:
        loss (Tensor). Calculation loss of victim detector
    """
    adv_images_batch_t = adv_images_batch_t.cuda()

    
    # forward the model
    with torch.set_grad_enabled(True):
        loss = model.forward_train(img=adv_images_batch_t, gt_bboxes=vanish_bboxes_list, img_metas=img_metas, gt_labels=vanish_labels_list)
    
    # print('loss:', loss)
    # assert False


    # 求和
    loss_cls_stack = torch.stack(loss['loss_cls'])
    loss_cls = torch.sum(loss_cls_stack)
    loss_bbox_stack = torch.stack(loss['loss_bbox'])
    loss_bbox = torch.sum(loss_bbox_stack)
    loss = 1.0 * loss_cls + 1.0 * loss_bbox

    # requires grad
    loss.requires_grad_(True)
    return loss

if __name__ == "__main__":
    from detection_model import init_detector

    batch_size = 8
    img_list = []
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/rotated_retinanet/latest.pth'
    model = init_detector(config=DetectorCfgSource, checkpoint=DetectorCheckpoint, device='cuda:0')  # 这里从init_detector返回的model已经是.eval()模式的
    
