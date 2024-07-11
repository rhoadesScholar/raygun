import torch
from raygun.torch.models import FreezableModel
from raygun.utils import passing_locals
import torch.nn.functional as F


class CycleModel(FreezableModel):
    """A CycleGAN model for image-to-image translation.

    Args:
        netG1 (``torch.nn.Module``):
            A generator network that maps from domain A to domain B.

        netG2 (``torch.nn.Module``): 
            A generator network that maps from domain B to domain A.

        scale_factor_A (``tuple[int]``, optional): 
            The downsampling factor for domain A images.

        scale_factor_B (``tuple[int]``, optional): 
            The downsampling factor for domain B images.

        split (``bool``, optional): 
            Whether to split the cycle loss into two parts (forward and backward).

        **kwargs: Additional arguments to be passed to the superclass constructor.
    """
    def __init__(
                self,
                netG1,
                netG2,
                scale_factor_A=None,
                scale_factor_B=None,
                split=False,
                **kwargs
            ) -> None:
        output_arrays: list[str] = ["fake_B", "cycled_B", "fake_A", "cycled_A"]
        nets:list = [netG1, netG2]
        super().__init__(**passing_locals(locals()))

        self.cycle:bool = True
        self.crop_pad:tuple = None  # TODO: Determine if this is depreciated

    def sampling_bottleneck(self, array:torch.Tensor, scale_factor:tuple) -> torch.Tensor:
        """Performs sampling bottleneck operation on the input tensor to avoid checkerboard artifacts.

        Args:
            array (``torch.Tensor``):
                A tensor of shape (batch_size, channels, height, width) or
                (batch_size, channels, depth, height, width) depending on the dimensions of the input data.
            scale_factor (``tuple[int]``):
                A tuple of scale factor for downsampling and upsampling the tensor.

        Returns:
            ``torch.Tensor``:
                A tensor of the same shape as the input tensor with applied sampling bottleneck operation.
        """
        size:torch.Size = array.shape[-len(scale_factor) :]
        mode: str = {2: "bilinear", 3: "trilinear"}[len(size)]
        down = F.interpolate(
            array, scale_factor=scale_factor, mode=mode, align_corners=True
        )
        return F.interpolate(down, size=size, mode=mode, align_corners=True)

    def set_crop_pad(self, crop_pad:int, ndims:int) -> None:
        """Set crop pad for the model.
    
        Args:
            crop_pad (``integer``): 
                The amount to crop the input by.
            ndims (``integer``): 
                The number of dimensions to apply crop pad.
        """
        self.crop_pad:tuple = (slice(None, None, None),) * 2 + (
            slice(crop_pad, -crop_pad),
        ) * ndims

    def forward(self, real_A=None, real_B=None) -> tuple:
        """Forward pass of the CycleGAN model.

        Args:
            real_A (``torch.Tensor``, optional): 
                Input tensor for domain A. Default: None.
            real_B (``torch.Tensor``, optional): 
                Input tensor for domain B. Default: None.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]: 
                A tuple containing:
                    - fake_B (Tensor or None): The generated output for domain B. None if real_A is None.
                    - cycled_B (Tensor or None): The generated output for domain A from the fake_B image. None if real_A is None or cycle is False.
                    - fake_A (Tensor or None): The generated output for domain A. None if real_B is None.
                    - cycled_A (Tensor or None): The generated output for domain B from the fake_A image. None if real_B is None or cycle is False.
        """

        assert (
            real_A is not None or real_B is not None
        ), "Must have some real input to generate outputs)"

        if (
            real_A is not None
        ):  # allow calling for single direction pass (i.e. prediction)
            fake_B = self.netG1(real_A)
            if self.crop_pad is not None:
                fake_B = fake_B[self.crop_pad]
            if self.scale_factor_B:
                fake_B: torch.Tensor = self.sampling_bottleneck(
                    fake_B, self.scale_factor_B
                )  # apply sampling bottleneck
            if self.cycle:
                if self.split:
                    cycled_A = self.netG2(
                        fake_B.detach()
                    )  # detach to prevent backprop to first generator
                else:
                    cycled_A = self.netG2(fake_B)
                if self.crop_pad is not None:
                    cycled_A = cycled_A[self.crop_pad]
            else:
                cycled_A = None
        else:
            fake_B = None
            cycled_A = None

        if real_B is not None:
            fake_A = self.netG2(real_B)
            if self.crop_pad is not None:
                fake_A = fake_A[self.crop_pad]
            if self.scale_factor_A:
                fake_A: torch.Tensor = self.sampling_bottleneck(
                    fake_A, self.scale_factor_A
                )  # apply sampling bottleneck
            if self.cycle:
                if self.split:
                    cycled_B = self.netG1(
                        fake_A.detach()
                    )  # detach to prevent backprop to first generator
                else:
                    cycled_B = self.netG1(fake_A)
                if self.crop_pad is not None:
                    cycled_B = cycled_B[self.crop_pad]
            else:
                cycled_B = None
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A
