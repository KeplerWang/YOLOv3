import os
import torch
import numpy as np
import shutil
from tqdm import tqdm
from matplotlib import patches, pyplot as plt
from torchvision.ops import batched_nms, box_convert, box_iou as iou
from utils.mAP import get_map


def iou_bbox(bbox1, bbox2, box_format1='xyxy', box_format2='xyxy'):
    assert box_format1 in ('xyxy', 'cxcywh', 'xywh'), 'box_format should be "xyxy, cxcywh, xywh"'
    assert box_format2 in ('xyxy', 'cxcywh', 'xywh'), 'box_format should be "xyxy, cxcywh, xywh"'
    bbox1 = box_convert(bbox1, box_format1, 'xyxy')
    bbox2 = box_convert(bbox2, box_format2, 'xyxy')
    return iou(bbox1, bbox2)


def iou_width_height(bbox1, bbox2):
    bbox1_w, bbox1_h = bbox1[0], bbox1[1]
    bbox2_w, bbox2_h = bbox2[:, 0], bbox2[:, 1]
    inter = torch.min(bbox1_w, bbox2_w) * torch.min(bbox1_h, bbox2_h)
    union = bbox1_w * bbox1_h + bbox2_w * bbox2_h - inter + 1e-16
    return inter / union


def prediction_transform(feature_map, anchors, image_input_size, is_pred=True):
    batch_size = feature_map.size()[0]
    cells_size = feature_map.size()[2]
    box_attrs = feature_map.size()[4]
    stride = image_input_size // cells_size
    anchors = anchors / stride

    # create the center offsets
    cell_len = torch.arange(cells_size)
    a, b = torch.meshgrid(cell_len, cell_len)
    y_offset = a.repeat(batch_size, 3, 1, 1).unsqueeze(-1)
    x_offset = b.repeat(batch_size, 3, 1, 1).unsqueeze(-1)
    x_y_offset = torch.cat((x_offset, y_offset), dim=-1).to(feature_map.device)

    if is_pred:
        # sigmoid the object confidence and centre_X, centre_Y
        feature_map[..., 0:3] = torch.sigmoid(feature_map[..., 0:3])
        # log space transform height and the width
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        feature_map[..., 3:5] = torch.exp(feature_map[..., 3:5]) * anchors
        # sigmoid the class scores
        feature_map[..., 5:] = torch.sigmoid(feature_map[..., 5:])

    # add the center offsets
    feature_map[..., 1:3] = feature_map[..., 1:3] + x_y_offset
    feature_map[..., 1:5] /= cells_size
    return feature_map.reshape(batch_size, -1, box_attrs)  # conf x y w h classes


def postprocessing(feature_maps, conf_threshold, nms_iou_threshold, is_pred=True):
    # feature_maps B x -1 x 25
    # transform x, y, w, h to x_min, y_min, x_max, y_max
    feature_maps[..., 3:5] = feature_maps[..., 1:3] + feature_maps[..., 3:5] / 2
    feature_maps[..., 1:3] = (feature_maps[..., 1:3] * 2 - feature_maps[..., 3:5]).clamp(0)

    output = [None] * feature_maps.size()[0]
    for idx, batch_fm in enumerate(feature_maps):
        if not is_pred:
            batch_fm_filtered = batch_fm[torch.eq(batch_fm[:, 0], 1)]
            output[idx] = torch.cat((
                batch_fm_filtered[:, :5], torch.ones_like(batch_fm_filtered[:, 5:6]), batch_fm_filtered[:, 5:]
            ), dim=1)
            continue
        # object confidence thresholding
        # get rid of confidence of bounding boxes less than threshold
        conf_cls, pred_cls = torch.max(batch_fm[:, 5:], dim=1, keepdim=True)
        conf_mask = torch.ge(batch_fm[:, 0] * conf_cls[:, 0], conf_threshold)
        batch_fm = batch_fm[conf_mask]
        conf_cls = conf_cls[conf_mask]
        pred_cls = pred_cls[conf_mask]
        if batch_fm.size()[0] == 0:
            continue
        batch_fm_filtered = torch.cat((batch_fm[:, :5], conf_cls, pred_cls), dim=1)
        keep_idx = batched_nms(
            boxes=batch_fm_filtered[:, 1:5],
            scores=batch_fm_filtered[:, 0] * batch_fm_filtered[:, 5],
            idxs=batch_fm_filtered[:, 6],
            iou_threshold=nms_iou_threshold
        )
        if keep_idx.size()[0] > 0:
            output[idx] = batch_fm_filtered[keep_idx]
    return output


def draw_image(image, save_path, boxes, classes):
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
    size = image.shape[0]
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        rect = patches.Rectangle(
            (box[1] * size, box[2] * size),
            (box[3] - box[1]) * size,
            (box[4] - box[2]) * size,
            linewidth=2,
            edgecolor=colors[int(box[-1])],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            box[1] * size,
            box[2] * size,
            s=f'{classes[int(box[-1])]} {round((box[0] * box[-2]).item(), 2)}',
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(box[-1])], "pad": 0},
        )
    plt.axis('off')
    plt.savefig(save_path)


def calc_mean_average_precision(
        model, loader,
        anchors,
        image_input_size,
        classes,
        mAP_out_path,
        conf_threshold, nms_iou_threshold, mAP_iou_threshold,
        verbose=False
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    anchors = torch.Tensor(anchors).to(device)
    ground_truth_path = os.path.join(mAP_out_path, 'ground_truth')
    detection_result_path = os.path.join(mAP_out_path, 'detection_results')
    if os.path.exists(detection_result_path):
        shutil.rmtree(detection_result_path)
    os.makedirs(detection_result_path)
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)
    
    model.eval()
    for x, y, names in tqdm(loader, desc='mAP'):
        x = x.to(device)
        with torch.no_grad():
            feature_maps = model(x)
        feature_maps = torch.cat([
            prediction_transform(feature_maps[i], anchors[i], image_input_size) for i in range(len(feature_maps))
        ], dim=1)
        y_boxes = prediction_transform(y[0].to(device), anchors[0], image_input_size, is_pred=False)
        y_boxes = postprocessing(y_boxes, conf_threshold, nms_iou_threshold, is_pred=False)
        pred_boxes = postprocessing(feature_maps, conf_threshold, nms_iou_threshold)
        for idx in range(x.size()[0]):
            y_box = y_boxes[idx]
            pred_box, name = pred_boxes[idx], names[idx]
            with open(os.path.join(detection_result_path, name + '.txt'), 'w') as f:
                if pred_box is not None:
                    for b in pred_box:
                        f.write('%s %s %s %s %s %s\n' % (
                            classes[b[-1].int().item()],
                            str((b[0] * b[-2]).item()),
                            str((b[1] * image_input_size).int().item()),
                            str((b[2] * image_input_size).int().item()),
                            str((b[3] * image_input_size).int().item()),
                            str((b[4] * image_input_size).int().item()),
                        ))
            with open(os.path.join(ground_truth_path, name + '.txt'), 'w') as f:
                if y_box is not None:
                    for b in y_box:
                        f.write('%s %s %s %s %s\n' % (
                            classes[b[-1].int().item()],
                            str((b[1] * image_input_size).int().item()),
                            str((b[2] * image_input_size).int().item()),
                            str((b[3] * image_input_size).int().item()),
                            str((b[4] * image_input_size).int().item())
                        ))
    return get_map(mAP_iou_threshold, verbose, mAP_out_path)


def calc_accuracy(model, loader, conf_threshold):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_class_pred, correct_class = 0, 0
    total_no_obj, correct_no_obj = 0, 0
    total_obj, correct_obj = 0, 0
    for idx, (x, y, _) in enumerate(tqdm(loader, desc='Acc')):
        x = x.to(device)
        with torch.no_grad():
            out = model(x)
        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 0] == 1
            no_obj = y[i][..., 0] == 0

            correct_class += torch.sum(
                torch.eq(torch.argmax(out[i][..., 5:][obj], dim=-1), y[i][..., 5][obj])
            )
            total_class_pred += torch.sum(obj)

            obj_pred = torch.gt(torch.sigmoid(out[i][..., 0]), conf_threshold)
            correct_obj += torch.sum(torch.eq(obj_pred[obj], y[i][..., 0][obj]))
            total_obj += torch.sum(obj)

            correct_no_obj += torch.sum(torch.eq(obj_pred[no_obj], y[i][..., 0][no_obj]))
            total_no_obj += torch.sum(no_obj)
    return (
        correct_class / (total_class_pred + 1e-16),
        correct_obj / (total_obj + 1e-16),
        correct_no_obj / (total_no_obj + 1e-16)
    )


# def test(model, loader):
#     model.eval()
#     anchors = torch.Tensor(config.anchors).to(config.device)
#     image_size = config.image_input_size
#
#     total_num = 0
#     total_cls_correct = 0
#     total_iou_gt_05 = 0
#     loop = tqdm(loader, desc='Acc')
#     for idx, (x, y) in enumerate(loop):
#         x = x.to(config.device)
#         with torch.no_grad():
#             predictions = model(x)
#         predictions_transformed = []
#         for i in range(3):
#             predictions_transformed.append(
#                 prediction_transform(predictions[i], anchors[i], image_size, is_pred=True)
#             )
#         pred_boxes = torch.Tensor([i[torch.sort(i[:, 0], descending=True)[1]][0].tolist()
#                                    for i in torch.cat(predictions_transformed, dim=1)]).to(config.device)
#         true_boxes = non_maximum_suppression(prediction_transform(
#             y[0].to(config.device), is_pred=False, anchors=torch.Tensor(config.anchors)[0]
#         ), conf_threshold=0.5, nms_iou_threshold=1, is_pred=False)[:, 1:]
#         mask = torch.eq(pred_boxes[:, -1], true_boxes[:, -1])
#         total_num += x.size()[0]
#         total_cls_correct += mask.sum()
#         pred_boxes_filtered = pred_boxes[mask]
#         true_boxes_filtered = true_boxes[mask]
#         ious = iou_bbox(pred_boxes_filtered, true_boxes_filtered, image_size, box_format='midpoint')
#         total_iou_gt_05 += torch.gt(ious, 0.5).sum()
#         loop.set_postfix(cls=total_cls_correct / total_num, iou=total_iou_gt_05 / total_num)
#     return total_cls_correct / total_num, total_iou_gt_05 / total_num


def load_checkpoints(model, path):
    print('=======> Loading CHECKPOINT:', path)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))


def save_checkpoints(model, path):
    print('=======> Saving CHECKPOINT:', path)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
