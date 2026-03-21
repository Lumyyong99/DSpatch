import mmcv
from mmcv.runner import load_checkpoint

from mmrotate.models import build_detector


def init_detector(config, checkpoint, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str): Config file path.
        checkpoint (str): Checkpoint path. 
        device (str): load model to GPU

    Returns:
        nn.Module: The constructed detector.
    """
    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    
    # Convert the model into evaluation mode
    model.eval()
    return model

if __name__ == '__main__':
    # Choose to use a config and initialize the detector
    config = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_training/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90.py'
    # Setup a checkpoint file to load
    checkpoint = '/home/yyx/Adversarial/mmrotate-0.3.3//UAV-ROD_training/detection_model/experiment2/epoch_15.pth'

    # Set the device to be used for evaluation
    device='cuda:0'

    # generate detector
    init_detector(config=config, checkpoint=checkpoint, device=device)