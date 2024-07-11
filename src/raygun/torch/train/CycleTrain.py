import gunpowder as gp

from raygun.torch.train import BaseTrain
from raygun.utils import passing_locals


class CycleTrain(BaseTrain):
    """CycleTrain object, an extension of BaseTrain for CycleGANs and CycleGAN-like models.
        
    Args:
        datapipes (``dict``): 
            A dictionary containing data pipelines.

        batch_request (``gunpowder.BatchRequest``): 
            The batch request to make to the pipeline.

        model (``torch.nn.Module``): 
            The model to use for training or testing.

        loss (``torch.nn.Module``): 
            The loss function to use.

        optimizer (``torch.nn.Module``): 
            The optimizer to use for training.
        tensorboard_path (``string``): 
            Path to store tensorboard files. Default is "./tensorboard/".

        log_every (``integer``): 
            Logging frequency. Default is 20.

        checkpoint_basename (``string``): 
            Base name of the file to store checkpoints. Default is "./models/model".

        save_every (``integer``): 
            Frequency of checkpoint saving. Default is 2000.

        spawn_subprocess (``bool``): 
            Whether or not to spawn a subprocess. Default is False.
        
        num_workers (``integer``): 
            Number of workers to use for parallelization. Default is 11.

        cache_size (``integer``): 
            Size of the cache. Default is 50.
            
        **kwargs: 
            Additional keyword arguments.
    """

    def __init__(
        self,
        datapipes: dict,
        batch_request: gp.BatchRequest,
        model,
        loss,
        optimizer,
        tensorboard_path: str = "./tensorboard/",
        log_every: int = 20,
        checkpoint_basename: str = "./models/model",
        save_every: int = 2000,
        spawn_subprocess: bool = False,
        num_workers: int = 11,
        cache_size: int = 50,
        **kwargs
    ):
        super().__init__(**passing_locals(locals()))

    def postnet_pipe(self, mode: str = "train") -> list:
        """Returns a list of postnet pipeline objects.
        
        Args:
            mode (``string``): 
                The mode to use for the pipeline. Default is "train".
        
        Returns:
            ``list``: 
                A list of postnet pipeline objects.
        """
        
        if mode == "test":
            stack = lambda dp: 1
        else:
            stack = lambda dp: dp.batch_size

        return [dp.postnet_pipe(batch_size=stack(dp)) for dp in self.datapipes.values()]
