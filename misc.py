import torch
from torch.nn.modules.pixelshuffle import PixelUnshuffle

def space_to_depth(tensor, scale_factor):
    """

    :param tensor:
    :param scale_factor:
    :return:
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and width of tensor must be divisible by scale_factor.')
    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    tensor = tensor.reshape([num, ch, new_height, scale_factor, new_width, scale_factor])  # divide by 2 the height and width then create four stacks
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    # tensor.permute returns a view of the original tensor with its dimensions permuted.
    # tensor.permute rearranges the original tensor according to the desired ordering;
    # the size of the returned tensor remains the same as that of the original.
    tensor = tensor.reshape([num, scale_factor * scale_factor, ch, new_height, new_width])
    return tensor


def main():
    x = torch.rand((2, 4, 8, 16))
    y = space_to_depth(x, scale_factor=2)
    print("old shape is{}. new shape is{}".format(x.shape, y.shape))
    s2d = PixelUnshuffle(downscale_factor=2)
    y2 = s2d(x)
    print("old shape is{}. new shape is{}".format(x.shape, y2.shape))
    print(y, y2)



if __name__ == "__main__":
    main()