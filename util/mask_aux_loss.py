import torch
import torch.nn.functional as F

def compute_normalized_embeddings(features, masks):
    """
    Args:
        features: Tensor of shape (batch_size, num_channels, H, W)
        masks: Tensor of shape (batch_size, num_instances, H, W), binary masks
    
    Returns:
        norm_embeddings: Normalized embeddings per instance
        instance_embeddings: Embeddings for each instance
    """
    batch_size, num_channels, H, W = features.shape
    _, num_instances, _, _ = masks.shape
    
    norm_embeddings = features / (features.sum(dim=(2, 3), keepdim=True) + 1e-6)
    
    instance_embeddings = torch.einsum('bchw,bkhw->bkch', norm_embeddings, masks)
    instance_masks_sum = masks.sum(dim=(2, 3), keepdim=True) + 1e-6  
    instance_embeddings /= instance_masks_sum

    return norm_embeddings, instance_embeddings

def pixel_wise_contrastive_loss(features, instance_embeddings, masks, temperature=0.1):
    """
    Args:
        features: Tensor of shape (batch_size, num_channels, H, W)
        instance_embeddings: Tensor of shape (batch_size, num_instances, num_channels)
        masks: Tensor of shape (batch_size, num_instances, H, W)
        temperature: A scalar for scaling during contrastive loss calculation
    
    Returns:
        loss: Scalar tensor of the total contrastive loss
    """
    batch_size, num_channels, H, W = features.shape
    _, num_instances, _, _ = masks.shape

    pixel_embeddings = features.view(batch_size, num_channels, H * W).permute(0, 2, 1)  # (batch_size, H*W, num_channels)
    instance_embeddings = instance_embeddings.view(batch_size, num_instances, num_channels)
    
    # Cosine similarity
    logits = torch.einsum('bnc,bmc->bnm', pixel_embeddings, instance_embeddings) / temperature
    labels = masks.view(batch_size, num_instances, H * W).permute(0, 2, 1)  # (batch_size, H*W, num_instances)

    loss = F.cross_entropy(logits, labels.argmax(dim=-1), reduction='mean')
    
    return loss
