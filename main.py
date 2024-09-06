# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------



import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import torch.nn as nn
from util.motdet_eval import motdet_evaluate, detmotdet_evaluate
from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, train_one_epoch_mot
from models import build_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from util.set_epoch import CustomSubset
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    #################################################################################
    # () Adding segmentation lr
    # parser.add_argument('--lr_segmentation_bbox_attention_names', default=["bbox_attention", "mask_head"], type=str, nargs='+')
    # parser.add_argument('--lr_segmentation_head', default=1e-4, type=float) # This value with SGD will prevent vanishing gradient descent but still the bbox_attention gradients are too small
    
    # parser.add_argument('--lr_segmentation_bbox_attention_names', default=["bbox_attention"], type=str, nargs='+')
    # parser.add_argument('--lr_segmentation_bbox_attention', default=2e-4, type=float)
    
    # parser.add_argument('--lr_segmentation_mask_head_names', default=["mask_head"], type=str, nargs='+')
    # parser.add_argument('--lr_segmentation_mask_head', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_bbox_attention_k_linear_names', default=["bbox_attention.k_linear"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_bbox_attention_k_linear', default=2e-4, type=float)
    parser.add_argument('--lr_segmentation_bbox_attention_q_linear_names', default=["bbox_attention.q_linear"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_bbox_attention_q_linear', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_lay1_names', default=["mask_head.lay1"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_lay1', default=2e-4, type=float)
    parser.add_argument('--lr_segmentation_mask_head_gn1_names', default=["mask_head.gn1"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_gn1', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_lay2_names', default=["mask_head.lay2"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_lay2', default=2e-4, type=float)
    parser.add_argument('--lr_segmentation_mask_head_gn2_names', default=["mask_head.gn2"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_gn2', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_lay3_names', default=["mask_head.lay3"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_lay3', default=2e-4, type=float)
    parser.add_argument('--lr_segmentation_mask_head_gn3_names', default=["mask_head.gn3"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_gn3', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_lay4_names', default=["mask_head.lay4"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_lay4', default=2e-4, type=float)
    parser.add_argument('--lr_segmentation_mask_head_gn4_names', default=["mask_head.gn4"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_gn4', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_lay5_names', default=["mask_head.lay5"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_lay5', default=2e-4, type=float)
    parser.add_argument('--lr_segmentation_mask_head_gn5_names', default=["mask_head.gn5"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_gn5', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_embedding_names', default=["mask_head.embedding"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_embedding', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_adapter1_names', default=["mask_head.adapter1"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_adapter1', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_adapter2_names', default=["mask_head.adapter2"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_adapter2', default=2e-4, type=float)

    parser.add_argument('--lr_segmentation_mask_head_adapter3_names', default=["mask_head.adapter3"], type=str, nargs='+')
    parser.add_argument('--lr_segmentation_mask_head_adapter3', default=2e-4, type=float)
    
    parser.add_argument('--lr_PerPixelEmbedding_names', default=["PerPixelEmbedding"], type=str, nargs='+')
    parser.add_argument('--lr_PerPixelEmbedding', default=2e-4, type=float)

    parser.add_argument('--lr_seg_branches_names', default=["seg_branches"], type=str, nargs='+')
    parser.add_argument('--lr_seg_branches', default=2e-4, type=float)
    
    parser.add_argument('--lr_AxialBlock_names', default=["AxialBlock"], type=str, nargs='+')
    parser.add_argument('--lr_AxialBlock', default=1e-3, type=float)
    
    parser.add_argument('--lr_FPNEncoder_names', default=["FPNEncoder"], type=str, nargs='+')
    parser.add_argument('--lr_FPNEncoder', default=1e-3, type=float)
    
    parser.add_argument('--lr_mask_positional_encoding_names', default=["mask_positional_encoding"], type=str, nargs='+')
    parser.add_argument('--lr_mask_positional_encoding', default=1e-3, type=float)
    
    # parser.add_argument('--lr_spatial_mlp_names', default=["spatial_mlp"], type=str, nargs='+')
    # parser.add_argument('--lr_spatial_mlp', default=1e-3, type=float)
    
    parser.add_argument('--lr_pos_cross_attention_names', default=["pos_cross_attention"], type=str, nargs='+')
    parser.add_argument('--lr_pos_cross_attention', default=1e-3, type=float)
    
    parser.add_argument('--lr_mask_embed_names', default=["mask_embed"], type=str, nargs='+')
    parser.add_argument('--lr_mask_embed', default=1e-3, type=float)
    
    #################################################################################
    parser.add_argument('--batch_size', default=2, type=int)
    ##################################################################################
    # () Changing weight decay
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--main_weight_decay', default=1e-4, type=float)
    # parser.add_argument('--backbone_weight_decay', default=1e-4, type=float)
    # parser.add_argument('--linear_proj_mult_weight_decay', default=1e-4, type=float)
    # parser.add_argument('--bbox_attention_weight_decay', default=5e-5, type=float)
    # parser.add_argument('--mask_head_weight_decay', default=1e-4, type=float)
    ##################################################################################
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_period', default=2, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--meta_arch', default='deformable_detr', type=str)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_anchors', default=1, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--enable_fpn', action='store_true')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--cj', action='store_true')
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--loss_normalizer', action='store_true')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--mix_match', action='store_true',)
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    ############################################################################
    # () Adding args for matcher
    # parser.add_argument('--set_cost_giou_mask_to_box', default=2, type=float,
    #                     help="giou mask coefficient in the matching cost")
    # parser.add_argument('--set_cost_mask_dice', default=2, type=float,
    #                     help="Dice loss mask coefficient in the matching cost")
    # parser.add_argument('--set_cost_mask_focal', default=2, type=float,
    #                     help="Focal loss mask coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=1, type=float,
                        help="Focal mask loss coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=2, type=float,
                        help="IOU mask loss coefficient in the matching cost")
    ############################################################################

    # * Loss coefficients
    ###########################################################################
    # () Adding masks loss coef
    parser.add_argument('--mask_loss_coef', default=5, type=float)
    parser.add_argument('--dice_loss_coef', default=3, type=float)
    parser.add_argument('--cost_giou_mask_to_box_coef', default=2, type=float)
    ###########################################################################
    parser.add_argument('--cls_loss_coef', default=3, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    # parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_alpha', default=1.25, type=float)
    parser.add_argument('--ae_loss_coef', default=2, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--gt_file_train', type=str)
    parser.add_argument('--gt_file_val', type=str)
    parser.add_argument('--coco_path', default='/data/workspace/detectron2/datasets/coco/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # end-to-end mot settings.
    parser.add_argument('--mot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--save_path', default='/output', type=str)
    parser.add_argument('--log_path', default='/outputs', type=str)
    parser.add_argument('--input_video', default='figs/demo.mp4', type=str)
    # parser.add_argument('--data_txt_path_train',
    #                     default='./datasets/data_path/detmot17.train', type=str,
    #                     help="path to dataset txt split")
    # parser.add_argument('--data_txt_path_val',
    #                     default='./datasets/data_path/detmot17.train', type=str,
    #                     help="path to dataset txt split")
    
    parser.add_argument('--data_txt_path_train',
                        default='./datasets/data_path/mots.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--data_txt_path_val',
                        default='./datasets/data_path/mots.train', type=str,
                        help="path to dataset txt split")
    
    # parser.add_argument('--data_txt_path_train',
    #                     default='./datasets/data_path/applemots.train', type=str,
    #                     help="path to dataset txt split")
    # parser.add_argument('--data_txt_path_val',
    #                     default='./datasets/data_path/applemots.val', type=str,
    #                     help="path to dataset txt split")
    
    # parser.add_argument('--data_txt_path_train',
    #                     default='./datasets/data_path/KITTImots.train', type=str,
    #                     help="path to dataset txt split")
    # parser.add_argument('--data_txt_path_val',
    #                     default='./datasets/data_path/KITTImots.val', type=str,
    #                     help="path to dataset txt split")
    
    parser.add_argument('--img_path', default='data/valid/JPEGImages/')

    parser.add_argument('--query_interaction_layer', default='QIM', type=str,
                        help="")
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--memory_bank_score_thresh', type=float, default=0.)
    parser.add_argument('--memory_bank_len', type=int, default=4)
    parser.add_argument('--memory_bank_type', type=str, default=None)
    parser.add_argument('--memory_bank_with_self_attn', action='store_true', default=False)

    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    
    # DN
    parser.add_argument('--dn', default='yes', type=str,
                        help="")
    
    # Dataloader observation
    parser.add_argument('--play', action='store_true', help='Enable play mode for visualization')

    return parser


def main(args):
    # writer = SummaryWriter(log_dir="/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/logs_norm/logs_loss")
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # model, criterion, postprocessors = build_model(args)
    # model.to(device)
    # output_dir = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR-mask-DN-DAB-Track-MOTS/outputs/model_two_stages.txt"
    # with open (output_dir, 'w') as f:
    #     f.write (str(model))

    # model_without_ddp = model
    
    # Freeze all parameters
    # for param in model_without_ddp.parameters():
    #     param.requires_grad = False

    # # Unfreeze segmentation head parameters
    # for param in model_without_ddp.PerPixelEmbedding.parameters():
    #     param.requires_grad = True

    # for param in model_without_ddp.AxialBlock.parameters():
    #     param.requires_grad = True
        
    # for param in model_without_ddp.transformer.pos_cross_attention.parameters():
    #     param.requires_grad = True
        
    # for param in model_without_ddp.transformer.parameters():
    #     param.requires_grad = True
        
        
    
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)

    # if args.distributed:
    #     if args.cache_mode:
    #         sampler_train = samplers.NodeDistributedSampler(dataset_train)
    #         sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
    #     else:
    #         sampler_train = samplers.DistributedSampler(dataset_train)
    #         sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)
    # if args.dataset_file in ['e2e_mot', 'e2e_dance', 'mot', 'ori_mot', 'e2e_static_mot', 'e2e_joint']:
    #     collate_fn = utils.mot_collate_fn
    # else:
    #     collate_fn = utils.collate_fn
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=collate_fn, num_workers=args.num_workers,
    #                                pin_memory=True)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
    #                              pin_memory=True)

    # def match_name_keywords(n, name_keywords):
    #     out = False
    #     for b in name_keywords:
    #         if b in n:
    #             out = True
    #             break
    #     return out
    
    # param_dicts = [
    #     {
    #         "params":
    #             [p for n, p in model_without_ddp.named_parameters()
    #              if not match_name_keywords(n, args.lr_backbone_names) 
    #              and not match_name_keywords(n, args.lr_linear_proj_names) 
                
    #              and not match_name_keywords(n, args.lr_PerPixelEmbedding_names)
    #              and not match_name_keywords(n, args.lr_AxialBlock_names)
    #              and not match_name_keywords(n, args.lr_pos_cross_attention_names)
    #              and not match_name_keywords(n, args.lr_mask_embed_names)
                
    #              and p.requires_grad],
    #         "lr": args.lr,
    #         # "weight_decay": args.main_weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() 
    #                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
    #         "lr": args.lr_backbone,
    #         # "weight_decay": args.backbone_weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() 
    #                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr * args.lr_linear_proj_mult,
    #         # "weight_decay": args.linear_proj_mult_weight_decay,
    #     },
       
    #     {
    #         "params": [
    #             p for n, p in model_without_ddp.named_parameters()
    #             if match_name_keywords(n, args.lr_PerPixelEmbedding_names) and p.requires_grad
    #         ],
    #         "lr": args.lr_PerPixelEmbedding,
    #         # "weight_decay": args.mask_head_weight_decay, 
    #     },


    #     {
    #         "params": [
    #             p for n, p in model_without_ddp.named_parameters()
    #             if match_name_keywords(n, args.lr_AxialBlock_names) and p.requires_grad
    #         ],
    #         "lr": args.lr_AxialBlock,
    #         # "weight_decay": args.mask_head_weight_decay, 
    #     },
        
        
    #     {
    #         "params": [
    #             p for n, p in model_without_ddp.transformer.named_parameters()
    #             if match_name_keywords(n, args.lr_pos_cross_attention_names) and p.requires_grad
    #         ],
    #         "lr": args.lr_pos_cross_attention,
    #         # "weight_decay": args.mask_head_weight_decay, 
    #     },
        
    #     {
    #         "params": [
    #             p for n, p in model_without_ddp.transformer.named_parameters()
    #             if match_name_keywords(n, args.lr_mask_embed_names) and p.requires_grad
    #         ],
    #         "lr": args.lr_mask_embed,
    #         # "weight_decay": args.mask_head_weight_decay, 
    #     },
        

        
    # ]
    
    # if args.sgd:
    #     optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
    #                                 weight_decay=args.weight_decay)
    # else:
    #     optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                                   weight_decay=args.weight_decay)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
   
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # if args.dataset_file == "coco_panoptic":
    #     # We also evaluate AP during panoptic training, on original coco DS
    #     coco_val = datasets.coco.build("val", args)
    #     base_ds = get_coco_api_from_dataset(coco_val)
    # else:
    #     base_ds = get_coco_api_from_dataset(dataset_val)

    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # if args.pretrained is not None:
    #     model_without_ddp = load_model(model_without_ddp, args.pretrained)
    
    # # if args.play:
    # #     visualization(args, data_loader_train, "train")
        
    # output_dir = Path(args.output_dir)
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    #     unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    #     if len(missing_keys) > 0:
    #         print('Missing Keys: {}'.format(missing_keys))
    #     if len(unexpected_keys) > 0:
    #         print('Unexpected Keys: {}'.format(unexpected_keys))
    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         import copy
    #         p_groups = copy.deepcopy(optimizer.param_groups)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         for pg, pg_old in zip(optimizer.param_groups, p_groups):
    #             pg['lr'] = pg_old['lr']
    #             pg['initial_lr'] = pg_old['initial_lr']
    #         # print(optimizer.param_groups)
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
    #         args.override_resumed_lr_drop = True
    #         if args.override_resumed_lr_drop:
    #             print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
    #             lr_scheduler.step_size = args.lr_drop
    #             lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    #         lr_scheduler.step(lr_scheduler.last_epoch)
    #         args.start_epoch = checkpoint['epoch'] + 1
    
    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return
    
    
    # K-Fold Cross Validation
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.dataset_file in ['e2e_mot', 'e2e_dance', 'mot', 'ori_mot', 'e2e_static_mot', 'e2e_joint']:
        collate_fn = utils.mot_collate_fn
    else:
        collate_fn = utils.collate_fn
    
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out   
    
    # Setup KFold
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
            
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)


    print("Start training")
    start_time = time.time()

    train_func = train_one_epoch
    if args.dataset_file in ['e2e_mot', 'e2e_dance', 'mot', 'ori_mot', 'e2e_static_mot', 'e2e_joint']:
        train_func = train_one_epoch_mot
        
        # Cross-validation loop
        fold = 0
        for fold, (train_index, val_index) in enumerate(kf.split(np.arange(len(dataset_train)))):
            print(f"Training on fold {fold+1}")
            
            writer = SummaryWriter(log_dir=f"/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/logs_AE/logs_loss/fold_{fold+1}")
    
            # Model initialization
            model, criterion, postprocessors = build_model(args)
            model.to(device)
            model_without_ddp = model
            
            output_dir = "/blue/hmedeiros/khademi.zahra/MOTR-train/MOTR_mask_AppleMOTS_train/MOTR_mask_DN_DAB/outputs/model_two_stages.txt"
            with open (output_dir, 'w') as f:
                f.write (str(model))
    
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('number of params:', n_parameters)
    
    
            # Subset train and validation datasets
            # dataset_train_fold = torch.utils.data.Subset(dataset_train, train_index)
            # dataset_val_fold = torch.utils.data.Subset(dataset_train, val_index)
            dataset_train_fold = CustomSubset(dataset_train, train_index)
            dataset_val_fold = CustomSubset(dataset_train, val_index)
            
            
            if args.distributed:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train_fold)
                    sampler_val = samplers.NodeDistributedSampler(dataset_val_fold, shuffle=False)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train_fold)
                    sampler_val = samplers.DistributedSampler(dataset_val_fold, shuffle=False)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train_fold)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val_fold)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
        
            
            data_loader_train = DataLoader(dataset_train_fold, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
            data_loader_val = DataLoader(dataset_val_fold, args.batch_size, sampler=sampler_val,
                                        drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                        pin_memory=True)
    
            # Defining optimizer inside the loop
            param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) 
                        and not match_name_keywords(n, args.lr_linear_proj_names) 
                        and not match_name_keywords(n, args.lr_PerPixelEmbedding_names)
                        and not match_name_keywords(n, args.lr_AxialBlock_names)
                        and not match_name_keywords(n, args.lr_pos_cross_attention_names)
                        and not match_name_keywords(n, args.lr_mask_embed_names)
                        and p.requires_grad],
                    "lr": args.lr,
                    # "weight_decay": args.main_weight_decay,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                    "lr": args.lr_backbone,
                    # "weight_decay": args.backbone_weight_decay,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                    "lr": args.lr * args.lr_linear_proj_mult,
                    # "weight_decay": args.linear_proj_mult_weight_decay,
                },
            
                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, args.lr_PerPixelEmbedding_names) and p.requires_grad
                    ],
                    "lr": args.lr_PerPixelEmbedding,
                    # "weight_decay": args.mask_head_weight_decay, 
                },


                {
                    "params": [
                        p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, args.lr_AxialBlock_names) and p.requires_grad
                    ],
                    "lr": args.lr_AxialBlock,
                    # "weight_decay": args.mask_head_weight_decay, 
                },
                
                
                {
                    "params": [
                        p for n, p in model_without_ddp.transformer.named_parameters()
                        if match_name_keywords(n, args.lr_pos_cross_attention_names) and p.requires_grad
                    ],
                    "lr": args.lr_pos_cross_attention,
                    # "weight_decay": args.mask_head_weight_decay, 
                },
                
                {
                    "params": [
                        p for n, p in model_without_ddp.transformer.named_parameters()
                        if match_name_keywords(n, args.lr_mask_embed_names) and p.requires_grad
                    ],
                    "lr": args.lr_mask_embed,
                    # "weight_decay": args.mask_head_weight_decay, 
                },   
            ]
            
            if args.sgd:
                optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                            weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                            weight_decay=args.weight_decay)
            
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
                model_without_ddp = model.module

            if args.dataset_file == "coco_panoptic":
                # We also evaluate AP during panoptic training, on original coco DS
                coco_val = datasets.coco.build("val", args)
                base_ds = get_coco_api_from_dataset(coco_val)
            else:
                base_ds = get_coco_api_from_dataset(dataset_val)

            if args.frozen_weights is not None:
                checkpoint = torch.load(args.frozen_weights, map_location='cpu')
                model_without_ddp.detr.load_state_dict(checkpoint['model'])

            if args.pretrained is not None:
                model_without_ddp = load_model(model_without_ddp, args.pretrained)
            
            output_dir = Path(args.output_dir)
            
            if args.resume:
                if args.resume.startswith('https'):
                    checkpoint = torch.hub.load_state_dict_from_url(
                        args.resume, map_location='cpu', check_hash=True)
                else:
                    checkpoint = torch.load(args.resume, map_location='cpu')
                missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
                if len(missing_keys) > 0:
                    print('Missing Keys: {}'.format(missing_keys))
                if len(unexpected_keys) > 0:
                    print('Unexpected Keys: {}'.format(unexpected_keys))
                if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                    import copy
                    p_groups = copy.deepcopy(optimizer.param_groups)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for pg, pg_old in zip(optimizer.param_groups, p_groups):
                        pg['lr'] = pg_old['lr']
                        pg['initial_lr'] = pg_old['initial_lr']
                    # print(optimizer.param_groups)
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                    args.override_resumed_lr_drop = True
                    if args.override_resumed_lr_drop:
                        print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                        lr_scheduler.step_size = args.lr_drop
                        lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                    lr_scheduler.step(lr_scheduler.last_epoch)
                    args.start_epoch = checkpoint['epoch'] + 1
            
            if args.eval:
                test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                    data_loader_val, base_ds, device, args.output_dir)
                if args.output_dir:
                    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                return
            
            dataset_train_fold.set_epoch(args.start_epoch)
            dataset_val_fold.set_epoch(args.start_epoch)
                
            
            # Start of training
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    sampler_train.set_epoch(epoch)
                train_stats = train_func(
                    model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)

                # () Logging
                for key, value in train_stats.items():
                    # writer.add_scalar(f'Training/{key}', value, epoch)
                    writer.add_scalar(f'Fold_{fold+1}/Training/{key}', value, epoch)
                
                lr_scheduler.step()
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'checkpoint.pth']
                    # extra checkpoint before LR drop and every 5 epochs
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_period == 0 or (((args.epochs >= 100 and (epoch + 1) > 100) or args.epochs < 100) and (epoch + 1) % 5 == 0):
                        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
                
                if args.dataset_file not in ['e2e_mot', 'e2e_dance', 'mot', 'ori_mot', 'e2e_static_mot', 'e2e_joint']:
                    test_stats, coco_evaluator = evaluate(
                        model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
                    )

                    # Log validation metrics
                    for key, value in test_stats.items():
                        # writer.add_scalar(f'Validation/{key}', value, epoch)
                        writer.add_scalar(f'Fold_{fold+1}/Validation/{key}', value, epoch)


                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters}

                    if args.output_dir and utils.is_main_process():
                        with (output_dir / "log.txt").open("a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                        # for evaluation logs
                        if coco_evaluator is not None:
                            (output_dir / 'eval').mkdir(exist_ok=True)
                            if "bbox" in coco_evaluator.coco_eval:
                                filenames = ['latest.pth']
                                if epoch % 50 == 0:
                                    filenames.append(f'{epoch:03}.pth')
                                for name in filenames:
                                    torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                            output_dir / "eval" / name)
                    
                    
                if args.dataset_file in ['e2e_mot', 'e2e_dance', 'mot', 'ori_mot', 'e2e_static_mot', 'e2e_joint']:
                    dataset_train.step_epoch()
                    dataset_val.step_epoch()

                # Log learning rate
                for i, group in enumerate(optimizer.param_groups):
                    # writer.add_scalar(f'Learning_Rate/group_{i}', group['lr'], epoch)
                    writer.add_scalar(f'Fold_{fold+1}/Learning_Rate/group_{i}', group['lr'], epoch)

                  
                writer.close()
                # pass
                
            # Reset model and optimizer for next fold
            del model, optimizer, lr_scheduler
            torch.cuda.empty_cache()
               
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)