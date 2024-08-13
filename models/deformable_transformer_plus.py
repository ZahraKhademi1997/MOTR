# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, masks_to_boxes
from models.ops.modules import MSDeformAttn

from util.bitmask import BitMasks

from util.DN import prepare_targets
from models.deformable_detr import MLP
from util.sineembed_position import gen_sineembed_for_position, _get_clones_dec
from util.axial_attention import AxialAttention, AxialBlock

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=True, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, dn = True, dn_num=300, noise_scale=0.4, num_classes = 1, initial_pred=True, learn_tgt = False, initialize_box_type = False, query_dim = 4, dec_layer_share = False):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.initialize_box_type = initialize_box_type
        
        # Defining denoising parameters
        self.dn=dn
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.num_classes = num_classes
        self.learn_tgt = learn_tgt 
        self.initial_pred = initial_pred
        self.decoder_norm = nn.LayerNorm(d_model)
        self._value = 300

        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(self.two_stage_num_proposals, d_model)
            
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(two_stage_num_proposals, 4)
            

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   num_feature_levels, nhead, dec_n_points, decoder_self_cross,
        #                                                   sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn)
        # self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, d_model, num_classes, return_intermediate_dec)
        
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross, 
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, self.decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          num_feature_levels=num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )
        self._bbox_embed = _bbox_embed = MLP(d_model, d_model, 4, 3)
        # nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        
        ## Xavier initialization instead of zero 
        # Initialize all layers except the last with Xavier 
        for layer in _bbox_embed.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Initialize the last layer with Xavier
        nn.init.xavier_uniform_(_bbox_embed.layers[-1].weight)
        nn.init.zeros_(_bbox_embed.layers[-1].bias)
        
        box_embed_layerlist = [_bbox_embed for i in range(num_decoder_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed
        
        self.class_embed = nn.Linear(d_model, num_classes)
        self.label_enc=nn.Embedding(num_classes,d_model)
        # Mask prediction
        self.mask_embed = MLP(d_model, d_model, d_model, 3)
        self.AxialBlock = AxialBlock(d_model,d_model // 2) # Initialized
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        #
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2) # To make it concatanable with DN queries
            
        self.num_decoder_layers = num_decoder_layers
        
        self._reset_parameters()

    
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
    #     N_, S_, C_ = memory.shape
    #     base_scale = 4.0
    #     proposals = []
    #     _cur = 0
    #     for lvl, (H_, W_) in enumerate(spatial_shapes):
    #         mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
    #         valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
    #         valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

    #         grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
    #                                         torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
    #         grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

    #         scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
    #         grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
    #         wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
    #         proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
    #         proposals.append(proposal)
    #         _cur += (H_ * W_)
    #     output_proposals = torch.cat(proposals, 1)
    #     output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    #     output_proposals = torch.log(output_proposals / (1 - output_proposals))
    #     output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    #     output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    #     output_memory = memory
    #     output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    #     output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    #     output_memory = self.enc_output_norm(self.enc_output(output_memory))
    #     return output_memory, output_proposals
    
    def gen_encoder_output_proposals(self, memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor): # Detection queries
        """
        Input:
            - memory: bs, \sum{hw}, d_model
            - memory_padding_mask: bs, \sum{hw}
            - spatial_shapes: nlevel, 2
        Output:
            - output_memory: bs, \sum{hw}, d_model
            - output_proposals: bs, \sum{hw}, 4
        """
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device)) # to get the central coordinates of the potentioal boxes
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2) # scale the grid to the memory dimension
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # Clamping to prevent inf or -inf
        output_proposals = torch.clamp(output_proposals, (1e-6), 1 - (1e-6))
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # normalizing the proposals = inverse sigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        return output_memory, output_proposals

    
    
    # Adding DN function
    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale
           
            known = [(torch.ones_like(targets.labels)).to(targets.labels.device)]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            # print('targets[boxes]:', targets.boxes.shape, 'scalalr:', scalar, 'targets[labels]:', targets.labels.shape)
            labels = targets.labels 
            boxes = targets.boxes 
            assert not torch.isnan(boxes).any(), "NaN values detected in target bounding boxes in prepare_for_dn."
            
            
            # Calculating min and max of the boxes to avoid small noised boxes
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            # Compute minimum dimensions as a fraction of the average dimensions
            min_width = widths.mean() * 0.1 # or torch.quantile(widths, 0.05) 
            min_height = heights.mean() * 0.1   
            
            # batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            batch_idx = torch.full_like(targets.labels.long(), 0) 
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).to(targets.labels.device) * noise_scale
                
                
                # Clamp the widths and heighdts to ensure they don't go below the minimum
                width_height = known_bbox_expand[:, 2:] - known_bbox_expand[:, :2]
                width_height = torch.stack([
                    torch.clamp(width_height[:, 0], min=min_width),
                    torch.clamp(width_height[:, 1], min=min_height)
                ], dim=1)
                known_bbox_expand[:, 2:] = known_bbox_expand[:, :2] + width_height
                
                # Correct any inversions
                x_min, y_min = known_bbox_expand[:, 0], known_bbox_expand[:, 1]
                x_max, y_max = known_bbox_expand[:, 2], known_bbox_expand[:, 3]
                known_bbox_expand[:, 0], known_bbox_expand[:, 2] = torch.min(x_min, x_max), torch.max(x_min, x_max)
                known_bbox_expand[:, 1], known_bbox_expand[:, 3] = torch.min(y_min, y_max), torch.max(y_min, y_max)
                
                # Clamping to be within the normalize range
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to(targets.labels.device)
            
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)
            # print('single_pad:', single_pad, 'pad_size:', pad_size)

            padding_label = torch.zeros(pad_size, self.d_model).to(targets.labels.device)
            padding_bbox = torch.zeros(pad_size, 4).to(targets.labels.device)
            assert not torch.isnan(padding_bbox).any(), "NaN values detected in padding_bbox in prepare_for_dn."

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to(targets.labels.device)
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.two_stage_num_proposals
            
            attn_mask = torch.ones(tgt_size, tgt_size).to(targets.labels.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def dn_post_process(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        assert not torch.isnan(outputs_coord).any(), "NaN values detected in outputs_coord in dn_post_process."
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask  
    
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    
    def forward(self, srcs, masks, pos_embeds, embeddings, targets = None, query_embed=None, ref_pts=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        memory, multi_level_outputs = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten) # For bboxes
        bs, _, c = memory.shape
        
        # Refining detection queries generated from encoder feature map
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes) # Generating proposals from the encoder output
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class = self.class_embed(output_memory)
            enc_outputs_coord = self._bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid # proposals coordinates
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
            reference_points = torch.gather(enc_outputs_coord, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid # retrieve features at the top k positions
            reference_points = reference_points.detach()
            assert not torch.isnan(reference_points).any(), "NaN values detected in inter_ref 415."
            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))  # unsigmoid # Content query initialization from the top k predictions

            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), srcs[0])
            tgt = tgt_undetach.detach() # preventing gradient flow from decoder to encoder
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = reference_points.sigmoid()
            assert not torch.isnan(interm_outputs['pred_boxes']).any(), "NaN values detected in interm_outputs[pred_boxes] in two stage."
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    reference_points = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(src_flatten.device)
                    assert not torch.isnan(reference_points).any(), "NaN values detected in inter_ref 436."  
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    reference_points = masks_to_boxes(flaten_mask > 0).to(src_flatten.device)
                    assert not torch.isnan(reference_points).any(), "NaN values detected in inter_ref 440."  
                else:
                    assert NotImplementedError
                
                # Adding sigmoid to prevent NAN in boxes and ref points  
                reference_points = box_xyxy_to_cxcywh(reference_points.sigmoid()) / torch.as_tensor([w, h, w, h],dtype=torch.float).to(src_flatten.device)                                                                        
                reference_points = reference_points.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                reference_points = inverse_sigmoid(reference_points) # Serves as the positional query
                # init_reference_out = reference_points
        else:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            reference_points = self.query_embed.weight[None].repeat(bs, 1, 1) # refpoint_embed in maskDINO
            assert not torch.isnan(reference_points).any(), "NaN values detected in inter_ref 451."
            # init_reference_out = reference_points
        
        
        # Adding DN training
        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, srcs[0].shape[0]) # tgt_mask = [tgt*tgt]
            assert not torch.isnan(input_query_bbox).any(), "NaN values detected in input_query_bbox."
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)
        
        # direct prediction from the matching and denoising part in the begining
        predictions_class = []
        predictions_mask = []
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), embeddings, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        if self.dn != "no" and self.training and mask_dict is not None:
            reference_points=torch.cat([input_query_bbox,reference_points],dim=1)
            
        
        assert not torch.isnan(reference_points).any(), "NaN values detected in reference_points general."
        assert not torch.isnan(reference_points.transpose(0, 1)).any(), "NaN values detected in reference_points transpose general."
        assert not torch.isnan(tgt).any(), "NaN values detected in tgt general."
        assert not torch.isnan(memory).any(), "NaN values detected in memory general."
        assert not torch.isnan(mask_flatten).any(), "NaN values detected in memory_key_padding_mask general."
        assert not torch.isnan(tgt_mask).any(), "NaN values detected in tgt_mask general."
       
        hs, inter_references = self.decoder(
            tgt=tgt.transpose(0, 1), 
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=reference_points.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )
        
            
        for inter_ref in inter_references:
            assert not torch.isnan(inter_ref).any(), "NaN values detected in inter_ref."
        
        inter_references_out = inter_references
        for inter_ref_out in inter_references_out:
            assert not torch.isnan(inter_ref_out).any(), "NaN values detected in inter_ref_out."
        
        # for i, output in enumerate(hs):
        #     outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), embeddings, self.training or (i == len(hs)-1))
        #     # outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), srcs[0], self.training or (i == len(hs)-1))
        #     predictions_class.append(outputs_class)
        #     predictions_mask.append(outputs_mask)

        # # iteratively box prediction
        # if self.initial_pred:
        #     out_boxes = self.pred_box(inter_references_out, hs, reference_points.sigmoid())
        #     assert not torch.isnan(out_boxes).any(), "NaN values detected in out_boxes in initial_pred if."
        #     assert len(predictions_class) == self.num_decoder_layers + 1
        # else:
        #     out_boxes = self.pred_box(inter_references_out, hs)
        #     assert not torch.isnan(out_boxes).any(), "NaN values detected in out_boxes in initial_pred else."
            
        # if mask_dict is not None:
        #     predictions_mask=torch.stack(predictions_mask)
        #     predictions_class=torch.stack(predictions_class)
        #     predictions_class, out_boxes,predictions_mask=\
        #         self.dn_post_process(predictions_class,out_boxes,mask_dict,predictions_mask) # Removing the denoising part
        #     # print('predictions_mask:', predictions_mask.shape) # torch.Size([7, 1, 300, 168, 233])
        #     predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
            
        # # elif self.training:  # this is to insure self.label_enc participate in the model
        # predictions_class[-1] = predictions_class[-1] + 0.0*self.label_enc.weight.sum()
        # self._value = predictions_class[-1].shape[1]
        # assert not torch.isnan(out_boxes[-1]).any(), "NaN values detected in out_boxes[-1]."
        
        # out = {
        #     'pred_logits': predictions_class[-1],
        #     'pred_masks': predictions_mask[-1],
        #     'pred_boxes':out_boxes[-1],
        #     'aux_outputs': self._set_aux_loss(
        #         predictions_class , predictions_mask, out_boxes
        #     )
        # }
        
        # if self.two_stage:
        #     out['interm_outputs'] = interm_outputs
            
        
        if self.two_stage:
            # return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact,memory
            # return hs, reference_points, inter_references_out, out, mask_dict
            return hs, reference_points, inter_references_out, mask_dict, predictions_class, predictions_mask, interm_outputs
        #######################################################################
        # (1) Adding memory to output in format that segmentation head expected
        # return hs, init_reference_out, inter_references_out, None, None
        # return hs, init_reference_out, inter_references_out, None, None, memory.permute(1, 2, 0).view(bs, c, h, w)
        # return hs, init_reference_out, inter_references_out, None, None, whole_memory 
        return hs, init_reference_out, inter_references_out, None, None, memory, multi_level_outputs
    #######################################################################
    
    def get_value(self):
        return self._value
    
    # def pred_box(self, reference, hs, ref0=None):
        
    #     """
    #     :param reference: reference box coordinates from each decoder layer
    #     :param hs: content
    #     :param ref0: whether there are prediction from the first layer
    #     """
    #     device = reference[0].device
        
    #     if ref0 is None:
    #         outputs_coord_list = []
    #     else:
    #         outputs_coord_list = [ref0.to(device)]
    #     for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
    #         assert not torch.isnan(layer_ref_sig).any(), "NaN values detected in layer_ref_sig in pred_box."
    #         assert not torch.isnan(layer_hs).any(), "NaN values detected in layer_hs in pred_box."

    #         layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
    #         assert not torch.isnan(inverse_sigmoid(layer_ref_sig)).any(), "NaN values detected in inverse_sigmoid(layer_ref_sig) in pred_box."
    #         layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
    #         layer_outputs_unsig = layer_outputs_unsig.sigmoid()
    #         assert not torch.isnan(layer_outputs_unsig).any(), "NaN values detected in layer_outputs_unsig in pred_box."
    #         outputs_coord_list.append(layer_outputs_unsig)
    #     outputs_coord_list = torch.stack(outputs_coord_list)
    #     return outputs_coord_list
    
    
    def forward_prediction_heads(self, output, mask_features, pred_mask=True): # Mask and class prediction head
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) # queries
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            # attention_embedding, similarity_h, similarity_w = self.AxialBlock(mask_features)
            # outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, attention_embedding)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return outputs_class, outputs_mask
    
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        
        multi_level_outputs = []
        for i, start_idx in enumerate(level_start_index):
            if i < len(level_start_index) - 1:
                end_idx = level_start_index[i + 1]
            else:
                end_idx = src.shape[1]
            multi_level_outputs.append(src[:, start_idx:end_idx])
            
        return src, multi_level_outputs


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output, multi_level_outputs = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output, multi_level_outputs


# class DeformableTransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False, extra_track_attn=False):
#         super().__init__()

#         self.self_cross = self_cross
#         self.num_head = n_heads

#         # cross attention
#         self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
#         # if self.use_deformable:
#         #     self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
#         # else:
#         #     self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout=dropout, activation=activation)
        
        
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)

#         # update track query_embed
#         self.extra_track_attn = extra_track_attn
#         if self.extra_track_attn:
#             print('Training with Extra Self Attention in Every Decoder.', flush=True)
#             self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#             self.dropout5 = nn.Dropout(dropout)
#             self.norm4 = nn.LayerNorm(d_model)

#         if self_cross:
#             print('Training with Self-Cross Attention.')
#         else:
#             print('Training with Cross-Self Attention.')

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
#         if self.extra_track_attn:
#             tgt = self._forward_track_attn(tgt, query_pos)

#         q = k = self.with_pos_embed(tgt, query_pos)
#         if attn_mask is not None:
#             tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
#                                   attn_mask=attn_mask)[0].transpose(0, 1)
#         else:
#             tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
#         tgt = tgt + self.dropout2(tgt2)
#         return self.norm2(tgt)

#     def _forward_track_attn(self, tgt, query_pos):
#         q = k = self.with_pos_embed(tgt, query_pos)
#         if q.shape[1] > 300:
#             tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
#                                     k[:, 300:].transpose(0, 1),
#                                     tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
#             tgt = torch.cat([tgt[:, :300],self.norm4(tgt[:, 300:]+self.dropout5(tgt2))], dim=1)
#         return tgt

#     def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
#                             src_padding_mask=None, attn_mask=None):

#         # self attention
#         tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
#         # cross attention
#         tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
#                                reference_points,
#                                src, src_spatial_shapes, level_start_index, src_padding_mask)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # ffn
#         tgt = self.forward_ffn(tgt)

#         return tgt

#     def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
#                             src_padding_mask=None, attn_mask=None):
#         # cross attention
#         tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
#                                reference_points,
#                                src, src_spatial_shapes, level_start_index, src_padding_mask)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         # self attention
#         tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
#         # ffn
#         tgt = self.forward_ffn(tgt)

#         return tgt

#     def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
#         attn_mask = None
#         if self.self_cross:
#             return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
#                                             level_start_index, src_padding_mask, attn_mask)
#         return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
#                                         src_padding_mask, attn_mask)


# class DeformableTransformerDecoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, d_model, num_classes, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.return_intermediate = return_intermediate
#         # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
#         # self.bbox_embed = None
#         # self.class_embed = None
#         self._bbox_embed = MLP(d_model, d_model, 4, 3)
#         self.class_embed = nn.Linear(d_model, num_classes)
#         self.label_enc=nn.Embedding(num_classes,d_model)
#         self.mask_embed = MLP(d_model, d_model, d_model, 3)

#     def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
#                 query_pos=None, src_padding_mask=None):
#         print('tgt:', tgt.shape, 'reference_points:', reference_points.shape, 'src:', src.shape, 'src_padding_mask:',src_padding_mask.shape)
#         output = tgt

#         intermediate = []
#         intermediate_reference_points = []
#         for lid, layer in enumerate(self.layers):
#             if reference_points.shape[-1] == 4:
#                 reference_points_input = reference_points[:, :, None] \
#                                          * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
#             else:
#                 assert reference_points.shape[-1] == 2
#                 reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
#             print('output:', output.shape)
#             output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

#             # hack implementation for iterative bounding box refinement
#             if self.bbox_embed is not None:
#                 tmp = self.bbox_embed[lid](output)
#                 if reference_points.shape[-1] == 4:
#                     new_reference_points = tmp + inverse_sigmoid(reference_points)
#                     new_reference_points = new_reference_points.sigmoid()
#                 else:
#                     assert reference_points.shape[-1] == 2
#                     new_reference_points = tmp
#                     new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
#                     new_reference_points = new_reference_points.sigmoid()
#                 reference_points = new_reference_points.detach()

#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_reference_points.append(reference_points)

#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(intermediate_reference_points)

#         return output, reference_points

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=True,
                 num_feature_levels=1,
                 deformable_decoder=True,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=True,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones_dec(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()
    
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,

                ):
        
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        
        # print("refpoints_unsigmoid stats - Min:", refpoints_unsigmoid.min(), "Max:", refpoints_unsigmoid.max())
        assert not torch.isnan(refpoints_unsigmoid).any(), "NaN found in refpoints_unsigmoid in decoder before any operation"
    
        output = tgt
        device = tgt.device

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        # print("reference_points stats - Min:", reference_points.min(), "Max:", reference_points.max())
        assert not torch.isnan(reference_points).any(), "NAN found in reference_points in decoder after sigmoid"
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)
            # print('reference_points:', reference_points.shape)
            reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2

            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,

                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,

                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask
            )

            # iter update
            if self.bbox_embed is not None:
                assert not torch.isnan(reference_points).any(), "NAN found in reference_points in decoder before"
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                assert not torch.isnan(reference_before_sigmoid).any(), "NAN found in reference_before_sigmoid in decoder"
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                assert not torch.isnan(new_reference_points).any(), "NAN found in new_reference_points in decoder"
                reference_points = new_reference_points.detach()
                assert not torch.isnan(reference_points).any(), "NAN found in reference_points in decoder"
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


class DeformableTransformerDecoderLayer(nn.Module):

    # def __init__(self, d_model=256, d_ffn=1024,
    #              dropout=0.1, activation="relu",
    #              n_levels=4, n_heads=8, n_points=4,
    #              use_deformable_box_attn=False,
    #              key_aware_type=None,
                #  ):
                     
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False, extra_track_attn=False, use_deformable_box_attn=False,
                 key_aware_type=None,):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')
        
        
        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
                                    k[:, 300:].transpose(0, 1),
                                    tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300],self.norm4(tgt[:, 300:]+self.dropout5(tgt2))], dim=1)
        return tgt

    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):
        
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
            
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                               reference_points.transpose(0, 1).contiguous(),
                               src.transpose(0, 1), src_spatial_shapes, level_start_index, src_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        attn_mask = None
        if self.self_cross:
            return self._forward_self_cross(tgt, tgt_query_pos, tgt_reference_points, memory, memory_spatial_shapes,
                                            memory_level_start_index, memory_key_padding_mask, attn_mask)
        return self._forward_cross_self(tgt, tgt_query_pos, tgt_reference_points, memory, memory_spatial_shapes,
                                            memory_level_start_index, memory_key_padding_mask, attn_mask)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        dn = True,
        dn_num=100, 
        noise_scale=0.4,
        num_classes = 1,
        initial_pred=True,
        learn_tgt = False,
        initialize_box_type = False,
        query_dim = 4,
        dec_layer_share = False,
    )






