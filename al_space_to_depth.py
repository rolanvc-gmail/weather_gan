import torch
import torch.nn.functional as F


def al_space_to_depth(tensor, scale_factor):
    """
    :param tensor:
    :param scale_factor:
    :return:
    """
    batch, num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and width of tensor must be divisible by scale_factor.')
    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    tensor = tensor.reshape([batch, num, ch, new_height, scale_factor, new_width, scale_factor])  # divide by 2 the height and width then create four stacks
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([batch, num, scale_factor*scale_factor, ch, new_height, new_width])
    return tensor


def space_to_depth_1(tensor, scale_factor):
    return F.pixel_unshuffle(tensor, scale_factor)


def test_space_to_depth():
    x = torch.rand((16, 1, 1, 256, 256))
    y = space_to_depth_1(x, scale_factor=2)
    assert y.shape == (16, 1, 4, 128, 128)


def main():
    test_space_to_depth()
    print("space_to_depth ok")


if __name__ == "__main__":
    main()