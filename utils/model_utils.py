import numpy as np

# Utility function to compute mask indices
def compute_mask_indices(
    shape,
    padding_mask,
    mask_prob,
    mask_length,
    mask_type="static",
    mask_other=0.0,
    min_masks=0,
    no_overlap=False,
    min_space=0,
):
    """
    Computes random mask spans for a given shape.
    NOTE: This function is taken from the original AV-HuBERT implementation.
    
    Args:
        shape: the shape for which to compute masks. 
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, 
            which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as the start of the span to be masked. 
            this will be multiplied by the number of timesteps divided by the length of the mask span
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: min space between spans (if no_overlap is True)
    
    Returns:
        A mask of shape (batch_size, sequence_length) where 1 indicates a masked position
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    # add a small number to avoid probs being 0
    mask_prob = mask_prob + 1e-5 
    
    all_num_mask = int(
        # add a small number to avoid rounding issues
        mask_prob * all_sz / float(mask_length)
        + 0.5
    )
    
    all_num_mask = max(min_masks, all_num_mask)
    
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a small number to avoid rounding issues
                mask_prob * sz / float(mask_length)
                + 0.5
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask
            
        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        else:
            raise NotImplementedError("Dynamic mask not implemented")
            
        if sum(lengths) == 0:
            lengths = np.array([0])
            num_mask = 1
            
        if no_overlap:
            mask_idc = []
            
            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space))
                if e - span_start - length - min_space >= keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
                
            parts = [(0, sz)]
            min_length = min(lengths)
            for i in range(num_mask):
                l = lengths[i] if i < len(lengths) else min_length
                lens = np.fromiter((e - s if e - s >= l + min_space else 0 for s, e in parts), dtype=np.int)
                lens_sum = lens.sum()
                if lens_sum == 0:
                    break
                probs = lens / lens_sum
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, l, min_length))
                
            mask_idc = np.array(mask_idc)
        else:
            # Find all possible mask indices
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
                
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            
            mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
            
        mask_idc = np.unique(mask_idc[mask_idc < sz])
        mask[i, mask_idc] = True
        
        mask_idcs.extend(mask_idc.tolist())
            
    return mask, None, None, None  # Return simplified version