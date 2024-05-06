


"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Module):
    
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr
        # self.temp = temp

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        # (1) Changing [1024, 512, 256]
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
        
    # (2) Adding track_instances  to the forward function
    # def forward(self, samples: NestedTensor):
    def forward(self, samples: NestedTensor, track_instances):
        if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples['imgs'])
        
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()

        assert mask is not None
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            
            assert mask is not None

        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)# torch.Size([1, 256, 14, 24])
                else:
                    src = self.detr.input_proj[l](srcs[-1]) #torch.Size([1, 256, 14, 24])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)  
                pos.append(pos_l)
                
        # (3) Affecting track_instances in calculation
        # hs,  init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory = self.detr.transformer([srcs[3]], [masks[3]], [pos[3]],self.detr.query_embed.weight)
        if track_instances is not None:
            hs,  init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory = self.detr.transformer([srcs[3]], [masks[3]], [pos[3]],track_instances.query_pos)
        else:
            hs,  init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory = self.detr.transformer([srcs[3]], [masks[3]], [pos[3]],self.detr.query_embed.weight)
        
        hs = hs
        init_reference = init_reference
    
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        
        # FIXME h_boxes takes the last one computed, keep this in mind
        ##############################################################################
        # (4) Changing the memory shape to calculate masks
        bs, c, h, w = srcs[3].shape
        memory = memory.view(bs, c, h, w)
        ##############################################################################
        
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=masks[3])
        
        # print('bbox_mask in segmenattion has the shape of:', bbox_mask.shape) #torch.Size([1, 310, 8, 14, 24])
        seg_mask = self.mask_head(srcs[3], bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        
        outputs_seg_masks = seg_mask.view(bs, outputs_coord[-1].shape[1], seg_mask.shape[-2], seg_mask.shape[-1])
        # print('outputs_seg_masks in segmentation has the shape of:', outputs_seg_masks.shape) # torch.Size([1, 300, 88, 118])
        out["pred_masks"] = outputs_seg_masks
        return out




class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        
        # self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)
        # Add embedding
        hid_dim = 256
        self.embedding = nn.Sequential(
            nn.Conv2d(
                inter_dims[4],  # The number of input channels should match the number of output channels of the previous layer
                hid_dim,  # The number of output channels is the dimensionality of the embedding space
                kernel_size=3, 
                stride=1,  
                padding=1,  
            ),
            nn.GroupNorm(32, hid_dim),  
            nn.ReLU(inplace=True),
        )

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        
        def save_image(feature_map, layer_name):
            image_path = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-AppleMots-MaskFormer-Instance/output/pred_masks/mask_segmentation_py"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i in range(feature_map.size(0)):
                plt.imshow(feature_map[i, 0].detach().cpu().numpy(), cmap='gray')
                plt.title(f"{layer_name}_{i}")
                filename = f"{layer_name}_{i}_{timestamp}.png"
                plt.savefig(os.path.join(image_path, filename))
                plt.close()
                
        x = x
        bbox_mask = bbox_mask
        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
        
        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)
        # Removing the output layer and replacing it with embedding layer
        # x = self.out_lay(x)
        # x = F.relu(x)
        x = self.embedding(x)
        # print('x_embedding shape:', x.shape) # torch.Size([300, 16, 120, 188])
        # print('x_out_lay shape:', x.shape) # torch.Size([300, 1, 120, 188])
        # print('x:', x)
        # save_image(x, 'out_lay')
        x = self.avgpool(x)    # Apply global average pooling to each feature map
        x = x.flatten(2)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        def save_attention_maps(attention, output_dir, layer_name):
            for idx, attention_map in enumerate(attention.detach().cpu().numpy()):
                # Define timestamp inside the loop to get a unique one for each file
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig, axs = plt.subplots(1, self.num_heads, figsize=(20, 2))  # Change figsize as needed
                for head in range(self.num_heads):
                    ax = axs[head] if self.num_heads > 1 else axs
                    ax.imshow(attention_map[head], cmap='viridis', interpolation='nearest')
                    ax.axis('off')
                plt.tight_layout()
                fig.suptitle(f'After - Layer {layer_name} - Head Attention Maps for Query {idx}')
                plt.savefig(os.path.join(output_dir, f'attention_map_q{idx}_layer_{layer_name}.png'))
                plt.close()
                
                
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias) #torch.Size([1, 256, 46, 64])
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # print('kh is:', kh.shape) #Conv2D: torch.Size([1, 8, 32, 46, 64]), Conv1D:  torch.Size([1, 8, 32, 2944])
        # print('qh is:', qh.shape) #Cov2D: torch.Size([1, 300, 8, 32]), Conv1D: torch.Size([1, 300, 8, 32])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        # print('weight has the shape of:', weights.shape) #Conv2D: torch.Size([1, 300, 8, 46, 64]), Conv1D: torch.Size([1, 300, 8, 2944])

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        
        save_attention_maps(weights[:, 0], '/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-AppleMots-MaskFormer-Instance/output/attention_map', 'weights')
        
        weights = self.dropout(weights)
        return weights


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]
        
        
# (5) Modifying dice loss
def dice_loss(inputs, targets, size, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # eps = 1e-5
    inputs = inputs.sigmoid()
    inputs_flat = inputs.flatten(1)
    # targets_flat = targets.flatten(1)
    numerator = 2 * (inputs_flat * targets).sum(-1)
    denominator = inputs_flat.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    # intersection = (inputs_flat * targets_flat).sum(dim=1)
    # union = (inputs_flat ** 2.0).sum(dim=1) + (targets_flat ** 2.0).sum(dim=1) + eps
    # loss = 1. - (2 * intersection / union)

    # print('inputs:', inputs.shape)
    # print('targets:', targets.shape)
    original_h, original_w = size
    # # Convert tensors to numpy arrays
    output_dir = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask_AppleMOTS-InstanceAware/output/pred_masks/dice_loss"
    inputs_reshaped = inputs.view(-1, original_h, original_w)
    targets_reshaped = targets.view(-1, original_h, original_w)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the batch and save images
    for i in range(inputs_reshaped.shape[0]):
        if i == 1:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(inputs_reshaped[i].detach().cpu(), cmap='gray')
            ax[0].set_title('Predicted Mask')
            ax[0].axis('off')

            ax[1].imshow(targets_reshaped[i].detach().cpu(), cmap='gray')
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')

            # Use datetime to generate a unique identifier for this particular batch and epoch
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"mask_comparison_ep_idx{i}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

    return loss.sum() / num_boxes


    
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2,mean_in_dim1=False):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[None], to_tuple(size), mode="bilinear").squeeze(0)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds



