import albumentations as album
import torch
from albumentations.pytorch import ToTensorV2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
dataset = 'PASCAL_VOC'
image_dir = 'dataset/' + dataset + '/images'
label_dir = 'dataset/' + dataset + '/labels'
train_csv_path = 'dataset/' + dataset + '/train.csv'
test_csv_path = 'dataset/' + dataset + '/test.csv'
PASCAL_VOC_classes = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
tiny_vid_classes = ['bird', 'car', 'dog', 'lizard', 'turtle']
image_input_size = 416
num_classes = 20
batch_size = 64
learning_rate = 4e-4
weight_decay = 4e-5
num_epochs = 100
conf_threshold = 0.5
nms_iou_threshold = 0.5
mAP_iou_threshold = 0.5
load_model = False
save_model = False
checkpoint = f'model/pth/{dataset}/'
mAP_out_path = f'mAP/{dataset}/'
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_count = torch.cuda.device_count()
base_anchors = [
    [(116, 90), (156, 198), (373, 326)],
    [(30, 61), (62, 45), (59, 119)],
    [(10, 13), (16, 30), (33, 23)]
]
anchors = (torch.Tensor(base_anchors) / 416 * image_input_size).round().int().tolist()
scale = 1.1
train_transforms = album.Compose([
    album.LongestMaxSize(max_size=int(image_input_size * scale)),
    album.PadIfNeeded(
        min_height=int(image_input_size * scale),
        min_width=int(image_input_size * scale),
        border_mode=0
    ),
    album.RandomCrop(width=image_input_size, height=image_input_size),
    album.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
    album.OneOf([
        album.ShiftScaleRotate(rotate_limit=15, border_mode=0),
        album.Affine(shear=15)
    ], p=0.5),
    album.HorizontalFlip(p=0.5),
    album.Blur(p=0.1),
    album.CLAHE(p=0.1),
    album.Posterize(p=0.1),
    album.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ToTensorV2()
],
    bbox_params=album.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
)
test_transforms = album.Compose([
    album.LongestMaxSize(max_size=image_input_size),
    album.PadIfNeeded(
        min_height=image_input_size,
        min_width=image_input_size,
        border_mode=0
    ),
    album.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ToTensorV2()
],
    bbox_params=album.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
)






