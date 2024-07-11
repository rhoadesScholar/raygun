from raygun.evaluation.validate_affinities import run_validation
import torch
from torch.utils.tensorboard import SummaryWriter

from raygun.utils import passing_locals
from raygun.torch.losses import GANLoss


class BaseCompetentLoss(torch.nn.Module):
    """Base loss function, implemented in PyTorch.

        Args:
            **kwargs:
                Optional keyword arguments.
    """

    def __init__(self, **kwargs) -> None:
            super().__init__()
            kwargs: dict = passing_locals(locals())
            for key, value in kwargs.items():
                setattr(self, key, value)

            if hasattr(self, "gan_mode"):
                self.gan_loss: GANLoss = GANLoss(gan_mode=self.gan_mode)

            self.loss_dict: dict = {}
    
    def set_requires_grad(self, nets:list, requires_grad=False) -> None:
        """Sets requies_grad=False for all the networks to avoid unnecessary computations.
        
        Args:
            nets (``list[torch.nn.Module, ...]``):
                A list of networks.

            requires_grad (``bool``):
                Whether the networks require gradients or not.
        """
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def crop(self, x:torch.Tensor, shape:tuple) -> torch.Tensor:
        """Center-crop x to match spatial dimensions given by shape.
        
        Args:
            x (``torch.Tensor``):
                The tensor to center-crop.
            
            shape (``tuple``):
                The shape to match the crop to.
        
        Returns:
            ``torch.Tensor``:
                The center-cropped tensor to the spatial dimensions given.
        """

        x_target_size:tuple = x.size()[: -self.dims] + shape

        offset: tuple = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices: tuple = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def clamp_weights(self, net:torch.nn.Module, min=-0.01, max=0.01) -> None:
        """Clamp the weights of a given network.
        
        Args:
            net (``torch.nn.Module``):
                The network to clamp.
            
            min (``float``, optional):
                The minimum value to clamp network weights to.

            max (``float``, optional):
                The maximum value to clamp network weights to.
        """

        for module in net.model:
            if hasattr(module, "weight") and hasattr(module.weight, "data"):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def add_log(self, writer, step) -> None:
        """Add an additional log to the writer, containing loss values and image examples.

        Args:
            writer (``SummaryWriter``):
                The display writer to append the losses & images to.
            
            step (``int``):
                The current training step.
        """

        # add loss values
        for key, loss in self.loss_dict.items():
            writer.add_scalar(key, loss, step)

        # add loss input image examples
        for name, data in self.data_dict.items():
            if len(data.shape) > 3:  # pull out batch dimension if necessary
                img = data[0].squeeze()
            else:
                img = data.squeeze()

            if len(img.shape) == 3:
                mid = img.shape[0] // 2  # for 3D volume
                img = img[mid]

            if (
                (img.min() < 0) and (img.min() >= -1.0) and (img.max() <= 1.0)
            ):  # scale img to [0,1] if necessary
                img = (img * 0.5) + 0.5
            writer.add_image(name, img, global_step=step, dataformats="HW")

    def update_status(self, step) -> None:
        if hasattr(self, "validation_config") and (
            step % self.validation_config["validate_every"] == 0
        ):
            run_validation(self.validation_config, step)
