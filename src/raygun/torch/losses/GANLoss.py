# ORIGINALLY WRITTEN BY TRI NGUYEN (HARVARD, 2021)
import torch


class GANLoss(torch.nn.Module):
    """Define different GAN objectives. The GANLoss class abstracts away the need to create the target label tensor that has the same size as the input.

        Args:
            gan_mode (``string``):
                The type of GAN objective. It currently supports vanilla, lsgan, and wgangp.

            target_real_label (``float``, optional):
                Label for a real image, with a default of 1.0.

            target_fake_label (``float``, optional):
                Label for a fake image fake image, with a default of 0.0.

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
    """

    def __init__(self, gan_mode:str, target_real_label=1.0, target_fake_label=0.0) -> None:
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode: str = gan_mode
        if gan_mode == "lsgan":
            self.loss: torch.nn.MSELoss = torch.nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction:torch.Tensor, target_is_real:bool) -> torch.Tensor:
        """Create label tensors with the same size as the input.

        Args:
            prediction (``torch.Tensor``):
                Typically the prediction from a discriminator.

            target_is_real (``bool``):
                Boolean to determine the ground truth label is for real images or fake images.

        Returns:
            ``torch.Tensor``:
                A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction:torch.Tensor, target_is_real:bool) -> float:
        """Calculate loss given Discriminator's output and grount truth labels.
        Args:
            prediction (``torch.Tensor``):
                Typically the prediction output from a discriminator.

            target_is_real (``bool``):
                Boolean to determine the ground truth label is for real images or fake images.

        Returns:
            ``float``:
                The calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor: torch.Tensor = self.get_target_tensor(prediction, target_is_real)
            loss: float = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
