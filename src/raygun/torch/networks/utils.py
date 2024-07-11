# ORIGINALLY WRITTEN BY TRI NGUYEN (HARVARD, 2021)

import torch
from torch.nn import init


def get_norm_layers(net:torch.nn.Module) -> list:
    """Get a list of all normalization layers in the given module.

    Args:
        net (``torch.nn.Module``): 
            The module to search for normalization layers.

    Returns:
        ``list``: 
            A list of all normalization layers found in the module.
    """

    return [n for n in net.modules() if "norm" in type(n).__name__.lower()]


def get_running_norm_stats(net:torch.nn.Module) -> tuple:
    """Get the running means and variances of all normalization layers in the given module.

    Args:
        net (``torch.nn.Module``): 
            The module to compute running means and variances for.

    Returns:
        ``tuple``: 
            A tuple containing two tensors, the concatenated running means and running variances, respectively.
            If no normalization layer is found, returns None for both elements.
    """

    means: list = []
    vars: list = []

    try:
        norms: list = get_norm_layers(net)

        for norm in norms:
            means.append(norm.running_mean)
            vars.append(norm.running_var)

        means = torch.cat(means)
        vars = torch.cat(vars)

    except:
        return None, None

    return means, vars


def set_norm_mode(net:torch.nn.Module, mode="train") -> None:
    """Set the normalization mode of the given module.

    Args:
        net (``torch.nn.Module``): 
            The module to set the normalization mode for.

        mode (``string``): 
            The normalization mode to set. Can be one of "train", "eval", or "fix_norms".
    """

    if mode == "fix_norms":
        net.train()
        for m in net.modules():
            if "norm" in type(m).__name__.lower():
                m.eval()

    if mode == "train":
        net.train()

    if mode == "eval":
        net.eval()


def init_weights(net, init_type="normal", init_gain=0.02, nonlinearity="relu") -> None:
    """Initialize network weights.

    Args:
        net (``torch.nn.Module``):
            The network to be initialized.

        init_type (``string``):
            The name of an initialization method: normal | xavier | kaiming | orthogonal.

        init_gain (``float``):
            Scaling factor for normal, xavier and orthogonal.

    'Normal' is used in the original pix2pix and CycleGAN paper. Xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m) -> None:  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class NoiseBlock(torch.nn.Module):
    """Definies a block for producing and appending a feature map of gaussian noise with mean=0 and stdev=1"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Standard NoiseBlock forward pass on a data tensor."""

        shape: list = list(x.shape)
        shape[1] = 1  # only make one noise feature
        noise: torch.Tensor = torch.empty(shape, device=x.device).normal_()
        return torch.cat([x, noise.requires_grad_()], 1)


class ParameterizedNoiseBlock(torch.nn.Module):
    """Definies a block for producing and appending a feature map of gaussian noise with mean and stdev defined by the first two feature maps of the incoming tensor"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Standard ParameterizedNoiseBlock forward pass on a data tensor."""

        noise: torch.Tensor = torch.normal(x[:, 0, ...], torch.relu(x[:, 1, ...])).unsqueeze(1)
        return torch.cat([x, noise.requires_grad_()], 1)
