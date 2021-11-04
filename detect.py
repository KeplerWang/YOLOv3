import argparse
import os
import config
import torch
from tqdm import tqdm
from utils.utils import load_checkpoints, prediction_transform, postprocessing, draw_image
from model.yolov3 import YOLOv3
from torchvision.transforms import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def detect(model, image_input_size, anchors, classes, src_path, des_path, conf_threshold, nms_iou_threshold):
    img_names = os.listdir(src_path)
    tf = transforms.Compose([
        transforms.Resize((image_input_size, image_input_size)),
        transforms.ToTensor()
    ])
    for img_name in tqdm(img_names, desc='Detect:'):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            img_path = os.path.join(src_path, img_name)
            save_path = os.path.join(des_path, img_name)
            image = tf(Image.open(img_path)).unsqueeze(0).to(model.device)
            with torch.no_grad():
                feature_maps = model(image)
            feature_maps = torch.cat([
                prediction_transform(feature_maps[i], anchors[i], image_input_size) for i in range(len(feature_maps))
            ], dim=1)
            outputs = postprocessing(feature_maps, conf_threshold, nms_iou_threshold)
            image = (255 * image).squeeze().permute(1, 2, 0).int().cpu().numpy()
            if outputs[0] is not None:
                draw_image(image, save_path, outputs[0], classes)
            else:
                image.save(save_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Detect Images from a image folder.'
    )
    ap.add_argument('-model', '--model_path', type=str, default=config.checkpoint + 'model_best.pth',
                    help='Path to store the YOLO model path.')
    ap.add_argument('-src', '--source', required=True, type=str,
                    help='Path to store the images to be detected.')
    ap.add_argument('-des', '--destination', type=str, default='runs/detect/',
                    help='Path to save the detected images.')
    ap.add_argument('-size', '--image_input_size', required=True, type=int,
                    help='Image size of the images tot be detected.')
    ap.add_argument('-ct', '--confidence', type=float, default=config.conf_threshold,
                    help='Bounding box confidence threshold.')
    ap.add_argument('-nt', '--nms', type=float, default=config.nms_iou_threshold,
                    help='NMS IoU threshold.')
    args = ap.parse_args()

    if args.model_path.__contains__('tiny_vid'):
        num_classes = 5
        image_input_size = 128
        classes = config.tiny_vid_classes
    else:
        num_classes = 20
        image_input_size = 416
        classes = config.PASCAL_VOC_classes
    yolo = YOLOv3(num_classes)
    if config.device_count > 1:
        yolo = torch.nn.DataParallel(yolo)
    yolo.to(config.device)
    load_checkpoints(yolo, args.model_path)
    yolo.eval()
    anchors = (torch.Tensor(config.anchors) / config.image_input_size * args.image_input_size).round().int()
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    detect(yolo, image_input_size, anchors, classes, args.source, args.destination, args.confidence, args.nms)