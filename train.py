import config
import torch.backends.cudnn
from torch.utils.data import DataLoader
from model.yolov3 import YOLOv3
from tqdm import tqdm
from loss import YOLOLoss
from YOLOdataset import YOLODataset
from utils.utils import calc_mean_average_precision, save_checkpoints, calc_accuracy
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    train_dataset = YOLODataset(config.train_csv_path, config.image_dir, config.label_dir, config.anchors,
                                config.num_classes, config.image_input_size, config.train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = YOLODataset(config.test_csv_path, config.image_dir, config.label_dir, config.anchors,
                               config.num_classes, config.image_input_size, config.test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = YOLOv3(config.num_classes)
    if config.device_count > 1:
        model = torch.nn.DataParallel(model)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()

    scaled_anchors = (torch.Tensor(config.anchors) / torch.Tensor([32, 16, 8]).reshape(3, 1, 1)).to(config.device)
    best_map = .0
    for epoch in range(config.num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Train {epoch + 1}/{config.num_epochs}', leave=True)
        losses = []
        for batch_idx, (x, y, _) in enumerate(loop):
            x = x.to(config.device)
            y0, y1, y2 = y[0].to(config.device), y[1].to(config.device), y[2].to(config.device)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                        criterion(out[0], y0, scaled_anchors[0])
                        + criterion(out[1], y1, scaled_anchors[1])
                        + criterion(out[2], y2, scaled_anchors[1])
                        )
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)
            writer.add_scalar('train/mean_loss', mean_loss, epoch)

        if (epoch + 1) % 1 == 0:
            class_acc, obj_acc, no_obj_acc = calc_accuracy(model, test_loader, config.conf_threshold)
            writer.add_scalar('test/class_acc', class_acc.item(), epoch)
            writer.add_scalar('test/obj_acc', obj_acc.item(), epoch)
            writer.add_scalar('test/no_obj_acc', no_obj_acc.item(), epoch)
        
        if (epoch + 1) > 50 and (epoch + 1) % 20 == 0:
            # cls_correct, iou_gt_05 = test(model, test_loader)
            # writer.add_scalar('test/cls_correct', cls_correct.item(), epoch)
            # writer.add_scalar('test/iou_gt_05', iou_gt_05.item(), epoch)
            mAP = calc_mean_average_precision(
                model, test_loader,
                config.anchors,
                config.image_input_size,
                config.PASCAL_VOC_classes,
                config.mAP_out_path,
                config.conf_threshold,
                config.nms_iou_threshold,
                config.mAP_iou_threshold,
                True
            )
            writer.add_scalar('test/mAP', mAP, epoch)
            if mAP > best_map:
                best_map = mAP
                save_checkpoints(model, f'{config.checkpoint}model_best.pth')







