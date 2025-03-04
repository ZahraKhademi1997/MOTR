# from typing import Optional
# from torch import Tensor
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter, UninitializedParameter
# import torch.nn.init as init
# from models.position_encoding import PositionEmbeddingSine
# # from models.mmcv_utils.mask_position_encoding import RelSinePositionalEncoding

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# class CrossAttentionLayer(nn.Module):
#     def __init__(self, d_model, nhead, attn_dim, dropout=0.0,
#                  activation="relu", normalize_before=False, positional_encoding=None):
#         super().__init__()
        
#         self.reduce_dim = nn.Conv1d(d_model, attn_dim, kernel_size=1)
#         self.expand_dim = nn.Conv1d(attn_dim, d_model, kernel_size=1)
        
#         self.multihead_attn = nn.MultiheadAttention(d_model // 2, nhead, dropout=dropout)

#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before
        
#         # self.positional_encoding = positional_encoding

#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt, memory,
#                      memory_mask: Optional[Tensor] = None,
#                      memory_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None,
#                      query_pos: Optional[Tensor] = None):
        
#         # Apply 1x1 conv to reduce dimensions 
#         if tgt.dim() == 2:
#             tgt = tgt.unsqueeze(1)  # Add a channel dimension at position 1
#         if memory.dim() == 2:
#             memory = memory.unsqueeze(1)
#         tgt = self.reduce_dim(tgt.permute(1, 2 , 0)).permute(2, 0 , 1)
#         memory = self.reduce_dim(memory.permute(1, 2 , 0)).permute(2, 0 , 1)
#         # query_pos = self.reduce_dim(query_pos.permute(1, 2 , 0)).permute(2, 0 , 1)

#         tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)
        
#         # Apply 1x1 conv to expand dimensions 
#         tgt2 = self.expand_dim(tgt2.permute(1, 0, 2)).permute(1, 0, 2)
        
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.norm(tgt)
#         tgt = self.norm(tgt)
#         # return tgt, attn_weights
#         return tgt

#     def forward_pre(self, tgt, memory,
#                     memory_mask: Optional[Tensor] = None,
#                     memory_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None):
        
#         # Apply 1x1 conv to reduce dimensions 
#         if tgt.dim() == 2:
#             tgt = tgt.unsqueeze(1)  # Add a channel dimension at position 1
#         if memory.dim() == 2:
#             memory = memory.unsqueeze(1)
#         tgt = self.reduce_dim(tgt.permute(1, 0, 2)).permute(1, 0, 2)  
#         memory = self.reduce_dim(memory.permute(1, 0, 2)).permute(1, 0, 2)
#         # query_pos = self.reduce_dim(query_pos.permute(1, 2 , 0)).permute(2, 0 , 1)
        
#         tgt2 = self.norm(tgt)
#         tgt2 , attn_map= self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)
        
#         # Apply 1x1 conv to expand dimensions 
#         tgt2 = self.expand_dim(tgt2.permute(1, 0, 2)).permute(1, 0, 2)
        
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.norm(tgt)
#         # return tgt, attn_map
#         return tgt


#     def forward(self, tgt, memory,
#                 memory_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(tgt, memory, memory_mask,
#                                     memory_key_padding_mask, pos, query_pos)
#         return self.forward_post(tgt, memory, memory_mask,
#                                  memory_key_padding_mask, pos, query_pos)


from typing import Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.init as init
from models.position_encoding import PositionEmbeddingSine
# from models.mmcv_utils.mask_position_encoding import RelSinePositionalEncoding

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, positional_encoding=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        # self.positional_encoding = positional_encoding

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
    
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        tgt = self.norm(tgt)
        # return tgt, attn_weights
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 , attn_map= self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        # return tgt, attn_map
        return tgt


    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


