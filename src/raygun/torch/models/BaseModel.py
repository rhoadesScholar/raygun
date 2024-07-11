import torch
from torch.utils.tensorboard import SummaryWriter

class BaseModel(torch.nn.Module):
    """Base class for bulding a PyTorch model.
    
    Args:
        **kwargs:
            Optional keyword arguments.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert hasattr(
            self, "output_arrays"
        ), "Model object must have list attribute `output_arrays` indicating what arrays are output by the model's forward pass, in order."

    def add_log(self, writer:SummaryWriter, iter:int) -> None:
        """Dummy model log add."""
        pass

    def forward(self) -> None:
        """Dummy forward pass."""
        return
