import torch

from torch.utils.tensorboard import SummaryWriter
from raygun.torch.models import BaseModel
from raygun.torch.networks.utils import *
from raygun.utils import passing_locals


class FreezableModel(BaseModel):
    """A base model class for torch models that can freeze normalization layers during training.

    Args:
        freeze_norms_at (``integer``): If set, normalization layers will be frozen
            after the given training step. Defaults to None.

        **kwargs: 
            Additional arguments to pass to the parent class.

    """

    def __init__(self, freeze_norms_at=None, **kwargs) -> None:
        super().__init__(**passing_locals(locals()))

    def set_norm_modes(self, mode:str = "train") -> None:
        """Set the mode for all normalization layers in the model.

        Args:
            mode (``string``): 
                The mode to set normalization layers to. Must be either "train" or "fix_norms".
        """

        for net in self.nets:
            set_norm_mode(net, mode)

    def add_log(self, writer:SummaryWriter, step:int) -> None:
        """Add histogram of the means and variances of the model's normalization layers to the
        given Tensorboard writer.

        Args:
            writer (``torch.utils.tensorboard.SummaryWriter``): 
                The Tensorboard writer.

            step (``integer``): 
                The current training step.
        """

        means: list = []
        vars: list = []
        for net in self.nets:
            mean, var = get_running_norm_stats(net)
            if mean is not None:
                means.append(mean)
                vars.append(var)

        if len(means) > 0:
            hists: dict[str: torch.Tensor] = {"means": torch.cat(means), "vars": torch.cat(vars)}
            for tag, values in hists.items():
                writer.add_histogram(tag, values, global_step=step)

    def update_status(self, step:int) -> None:
        """Update the status of the model to freeze the normalization layers, if applicable.

        Args:
            step (``integer``): 
                The current training step.
        """
        if self.freeze_norms_at is not None and step >= self.freeze_norms_at:
            self.set_norm_modes(mode="fix_norms")
