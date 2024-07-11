import torch


def get_base_optimizer(optim) -> torch.optim.Optimizer:
    """Return the base optimizer object given its name or object.

    Args:
        optim (``string``, torch.optim.Optimizer``): 
            String or optimizer object

    Returns:
        ``torch.optim.Optimizer``:
            The base optimizer object.
    """

    if isinstance(optim, str):
        base_optim: torch.optim.Optimizer = getattr(torch.optim, optim)
    else:
        base_optim = optim
    return base_optim
