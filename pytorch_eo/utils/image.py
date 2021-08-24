def pad_and_tile(tensor, window, stride):
    h, w = tensor.shape[-2:]
    wh, ww = window
    sh, sw = stride
    # compute padding
    pad_h = wh - (h % wh)
    pad_w = ww - (w % ww)
    # pad tensor [C, H, W]
    pad = (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2)
    pad_t = torch.nn.functional.pad(
        input=tensor, pad=pad, mode='constant', value=0)
    return pad_t.unfold(1, wh, sh).unfold(2, ww, sw).permute(1, 2, 0, 3, 4).contiguous(), pad


def untile(tensor, unfold_shape, pad):
    pad_w1, pad_w2, pad_h1, pad_h2 = pad
    # tensor [R, Co, C, H, W]
    output_h = unfold_shape[0] * unfold_shape[3]
    output_w = unfold_shape[1] * unfold_shape[4]
    tensor_orig = tensor.permute(2, 0, 3, 1, 4).contiguous()
    # remove padding
    return tensor_orig.view(-1, output_h, output_w)[:, pad_h1:-pad_h2, pad_w1:-pad_w2]
