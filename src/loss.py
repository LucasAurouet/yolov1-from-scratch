import torch
from .utils import ind_ij

def yolo_loss(pred, true, S, B, C, lambda_coord, lambda_noobj):
    # Computes the loss between the (N, S, S, C+5*B) true and predicted tensors
    
    def coord_loss(pred_coord, true_coord, ind, lambda_coord=5):
        # coordinates loss -> boxes location
        loss_xy = (pred_coord[..., :2] - true_coord[..., :2])**2
        loss_wh = (torch.sqrt(pred_coord[..., 2:]) - torch.sqrt(true_coord[..., 2:]))**2
        loss_xywh = torch.cat([loss_xy, loss_wh], dim=-1)
        # Apply indicator mask (only the responsible box)
        loss_xywh = loss_xywh * ind.unsqueeze(-1)
        
        return lambda_coord * torch.sum(loss_xywh)
    
    def conf_loss(pred_conf, ious, ind, lambda_noobj=0.5):
        # confidence loss -> object detection withihn each cell 
        loss_obj = torch.sum(ind * (pred_conf - ious)**2)
        loss_noobj = torch.sum((1 - ind) * pred_conf**2)

        return loss_obj + lambda_noobj * loss_noobj
    
    def class_loss(pred_classes, true_classes, obj_mask):
        # class loss -> classification of the object if any
        return torch.sum(obj_mask.unsqueeze(-1) * (pred_classes - true_classes) ** 2)
    
    # Find the batch size 
    if pred.dim()==3:
        batch_size = 1
        true = true.unsqueeze(0) 
    else:
        batch_size = pred.shape[0]
    
    # x, y, w, h, confidence
    # (S, S, B * 5 + C) -> (S, S, B, 5)
    true_boxes = true[..., 0:B * 5].reshape(batch_size, S, S, B, 5)
    pred_boxes = pred[..., 0:B * 5].reshape(batch_size, S, S, B, 5)
    # One-hot predicted classes
    # (S, S, B * 5 + C) -> (S, S, B, C)
    pred_classes = pred[..., B * 5:B * 5 + C]
    true_classes = true[..., B * 5:B * 5 + C]
    # Extract the confidence
    # (N, S, S, B)
    pred_conf = pred_boxes[..., 4] 
    true_conf = true_boxes[..., 4]
    # Extract the coordinates
    # (N, S, S, B, 4)
    pred_coord = pred_boxes[..., :4] 
    true_coord = true_boxes[..., :4]
    
    # Transform to avoid negative values and undefined roots 
    pred_xy = pred_coord[..., :2]
    # pred_wh = torch.exp(pred_coord[..., 2:4])
    pred_wh = torch.clamp(pred_coord[..., 2:4], min=1e-6)
    pred_coord = torch.cat([pred_xy, pred_wh], dim=-1)
    
    # compute IOUs and identify responsible boxes (largest iou)
    obj_mask = (true_conf[..., 0] > 0).float()  # (N, S, S)
    ind, ious = ind_ij(pred_coord, true_coord, obj_mask)
    
    total_loss = (
        coord_loss(pred_coord, true_coord, ind, lambda_coord)
        + class_loss(pred_classes, true_classes, obj_mask)
        + conf_loss(pred_conf, ious, ind, lambda_noobj)
    )
    print('coord : ', coord_loss(pred_coord, true_coord, ind, lambda_coord))
    print('class : ', class_loss(pred_classes, true_classes, obj_mask))
    print('conf : ', conf_loss(pred_conf, ious, ind, lambda_noobj))
    
    # Normalize by batch_size
    return total_loss / batch_size