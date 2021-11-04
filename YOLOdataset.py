import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from utils.utils import iou_width_height
ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
            self,
            annotations_csv_dir,
            image_dir,
            label_dir,
            anchors,
            num_classes,
            image_input_size,
            transform=None
    ):
        super(YOLODataset, self).__init__()
        self.annotations = pd.read_csv(annotations_csv_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.num_classes = num_classes
        self.image_input_size = image_input_size
        self.feature_map_scale = [image_input_size // 32, image_input_size // 16, image_input_size // 8]
        self.anchors = torch.Tensor(anchors).flatten(0, 1)
        self.ignore_iou_threshold = 0.5
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_ = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        label_ = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        image = Image.open(image_).convert('RGB')
        bounding_boxes = np.roll(np.loadtxt(fname=label_, delimiter=' ', ndmin=2), -1, axis=1)

        if self.transform is not None:
            augmentations = self.transform(image=np.array(image), bboxes=bounding_boxes)
            image = augmentations['image']
            bounding_boxes = augmentations['bboxes']
        else:
            image = transforms.ToTensor()(image)

        targets = [torch.zeros((3, s, s, 6)) for s in self.feature_map_scale]
        for bbox in bounding_boxes:
            # check which anchor has max IoU with bbox
            iou_anchors = iou_width_height(torch.Tensor(bbox[2:4]), self.anchors / self.image_input_size)
            iou_anchors_index = iou_anchors.argsort(descending=True)
            x, y, w, h, c = bbox
            has_anchor = [False] * 3
            for anchor_index in iou_anchors_index:
                # which scale does this anchor belong to
                scale_index = anchor_index // 3
                # what is the index of this anchor on this scale
                anchor_index_on_scale = anchor_index % 3
                # feature map size 13, 26, 52
                fm_scale = self.feature_map_scale[scale_index]
                # which cell does this bbox belong to
                cell_i_index, cell_j_index = int(fm_scale * y), int(fm_scale * x)
                # check whether there already exists an anchor (if exists, anchor_taken = 1)
                anchor_taken = targets[scale_index][anchor_index_on_scale, cell_i_index, cell_j_index, 0]
                if anchor_taken == 0 and not has_anchor[scale_index]:
                    targets[scale_index][anchor_index_on_scale, cell_i_index, cell_j_index, 0] = 1
                    # the relative x, y, w, h with cell
                    new_x, new_y = fm_scale * x - cell_j_index, fm_scale * y - cell_i_index
                    new_w, new_h = fm_scale * w, fm_scale * h
                    targets[scale_index][anchor_index_on_scale, cell_i_index, cell_j_index, 1:5] = torch.Tensor([
                        new_x, new_y, new_w, new_h
                    ])
                    targets[scale_index][anchor_index_on_scale, cell_i_index, cell_j_index, 5] = int(c)

                    has_anchor[scale_index] = True
                elif anchor_taken == 0 and iou_anchors[anchor_index] > self.ignore_iou_threshold:
                    # ignore
                    targets[scale_index][anchor_index_on_scale, cell_i_index, cell_j_index, 0] = -1
        return image, tuple(targets), self.annotations.iloc[idx, 0].split('.')[0]


if __name__ == "__main__":
    import config
    from torch.utils.data import DataLoader
    from utils.utils import prediction_transform, postprocessing, draw_image

    dataset = YOLODataset(
        config.train_csv_path,
        config.image_dir,
        config.label_dir,
        image_input_size=config.image_input_size,
        num_classes=config.num_classes,
        anchors=config.anchors,
        transform=config.test_transforms,
    )
    i = 0
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y, _ in loader:
        boxes = prediction_transform(
            y[0], image_input_size=config.image_input_size, is_pred=False, anchors=torch.Tensor(config.anchors)[0]
        )
        boxes = postprocessing(boxes, config.conf_threshold, config.nms_iou_threshold, is_pred=False)[0]
        draw_image(x[0].permute(1, 2, 0).to("cpu"), f'{i}.png', boxes, config.tiny_vid_classes)
        i += 1
        if i == 10:
            break
