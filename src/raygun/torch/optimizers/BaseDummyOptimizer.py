import itertools
import torch


class BaseDummyOptimizer(torch.nn.Module):
    """Base Dummy Optimizer for training.

    Args:
        scheduler (``string``, optional): 
            The name of the learning rate scheduler to use. Default is None.

        scheduler_kwargs (``dict``, optional): 
            A dictionary of keyword arguments to pass to the learning rate scheduler. Default is an empty dictionary.

        **optimizers (optional): 
            Keyword arguments for the optimizer(s) to use.
    """

    def __init__(self, scheduler=None, scheduler_kwargs={}, **optimizers) -> None:
        super().__init__()

        self.schedulers: dict = {}

        for name, optimizer in optimizers.items():
            setattr(self, name, optimizer)

            """
            For 'LambdaLR', we keep the same learning rate for the first <n_epochs> epochs
            and linearly decay the rate to zero over the next <n_epochs_decay> epochs.
            For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
            See https://pytorch.org/docs/stable/optim.html for more details.
            """

            if isinstance(scheduler, str):
                if scheduler == "LambdaLR":

                    def lambda_rule(epoch:int) -> float:
                        lr_l: float = 1.0 - max(
                            0,
                            epoch
                            + scheduler_kwargs["epoch_count"]
                            - scheduler_kwargs["n_epochs"],
                        ) / (scheduler_kwargs["n_epochs_decay"] + 1.0)
                        return lr_l

                    self.schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=lambda_rule
                    )

                else:
                    self.schedulers[name] = getattr(
                        torch.optim.lr_scheduler, scheduler
                    )(optimizer, **scheduler_kwargs)

            elif scheduler is not None:
                self.schedulers[name] = scheduler(optimizer, **scheduler_kwargs)

    def step(self) -> None:
        """Takes a step of the optimizer(s) and the corresponding scheduler(s). """
        for name, scheduler in self.schedulers.items():
            scheduler.step()
