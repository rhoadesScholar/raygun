import inspect
import os
import sys
from raygun.read_config import read_config
from tqdm import trange
import gunpowder as gp
import numpy as np

import logging

logger: logging.Logger = logging.getLogger(__name__)

from raygun.utils import passing_locals, to_json

class BaseTrain(object):
    """Base training class for models.

    Args:
        datapipes (``dict``): 
            Dictionary of Gunpowder datapipes.

        batch_request (``gunpowder.BatchRequest``): 
            Request to use when running Gunpowder.

        model (``torch.nn.Module``): 
            PyTorch model to use for training.

        loss (``torch.nn.Module``): 
            PyTorch loss function to use for training.

        optimizer (``torch.nn.Module``): 
            PyTorch optimizer to use for training.

        tensorboard_path (``string``, optional): 
            Path to use for Tensorboard logs. Defaults to "./tensorboard/".

        log_every (``integer``, optional): 
            How often to log loss during training. Defaults to 20.

        checkpoint_basename (``string``, optional): 
            Basename to use for model checkpoints. Defaults to "./models/model".

        save_every (``intger``, optional): 
            How often to save a model checkpoint. Defaults to 2000.

        spawn_subprocess (``bool``, optional): 
            Whether to spawn a subprocess to run Gunpowder. Defaults to False.

        num_workers (``integer``, optional): 
            Number of workers to use with the Gunpowder PreCache node. Defaults to 11.

        cache_size (``integer``, optional): 
            Cache size to use with the Gunpowder PreCache node. Defaults to 50.

        snapshot_every (``integer``, optional): 
            How often to save a snapshot of the training volumes. Defaults to None.
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
        snapshot_every=None,
        **kwargs,
    ) -> None:
        kwargs: dict = passing_locals(locals())
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.arrays: dict = {}
        for datapipe in datapipes.values():
            self.arrays.update(datapipe.arrays)

        self.input_dict: dict = {}
        for array_name in inspect.signature(model.forward).parameters.keys():
            if array_name != "self":
                self.input_dict[array_name] = self.arrays[array_name]

        self.output_dict: dict = {}
        for i, array_name in enumerate(model.output_arrays):
            if array_name not in self.arrays.keys():
                self.arrays[array_name] = gp.ArrayKey(array_name.upper())
            self.output_dict[i] = self.arrays[array_name]

        self.loss_input_dict: dict = {}
        for i, array_name in enumerate(
            inspect.signature(loss.forward).parameters.keys()
        ):
            if array_name != "self":
                if array_name not in self.arrays.keys():
                    self.arrays[array_name] = gp.ArrayKey(array_name.upper())
                self.loss_input_dict[i] = self.arrays[array_name]

        # create a train node using our model, loss, and optimizer
        self.train_node = gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs=self.input_dict,
            outputs=self.output_dict,
            loss_inputs=self.loss_input_dict,
            log_dir=tensorboard_path,
            log_every=log_every,
            checkpoint_basename=checkpoint_basename,
            save_every=save_every,
            spawn_subprocess=spawn_subprocess,
        )

        os.makedirs(os.path.dirname(checkpoint_basename), exist_ok=True)

    def prenet_pipe(self, mode: str = "train"):
        """Creates a pipeline that preprocesses the input data. The pre-processing pipeline is created by calling the prenet_pipe() method on all data pipes, and then merging the output streams into one using the MergeProvider() node.

        Args:
            mode (``string``, optional):
                The mode in which the data will be processed, defaults to "train."

        Returns:
            ``tuple``:
                A tuple that contains the pre-processed data from all data pipes merged using MergeProvider() node.
        """

        return (
            tuple([dp.prenet_pipe(mode) for dp in self.datapipes.values()])
            + gp.MergeProvider()
        )  # merge upstream pipelines for multiple sources

    def postnet_pipe(self, batch_size=1) -> list:
        """Creates a post-processing pipeline that is responsible for processing the output of the network.
        The pipeline is created by calling the postnet_pipe() method on all data pipes and storing the output streams in a list.

        Args:
            batch_size (``integer``, optional)
                The batch size of the data, with a default value of 1.

        Returns:
            ``list``:
                A list of post-processed data from all data pipes.
        """

        return [
            dp.postnet_pipe(batch_size=batch_size) for dp in self.datapipes.values()
        ]

    def training_pipe(self, mode: str = "train"):
        """Creates a pipeline for training the neural network. The pipeline is created by calling the prenet_pipe() method to create a pre-processing pipeline, adding a PreCache() node if mode is "train", adding the train_node to the pipeline, calling the postnet_pipe() method to create a post-processing pipeline, and adding a Snapshot() node if mode is "train" and snapshot_every is not None. 

        Args:
            mode (`str``, optional):
                The mode in which the data will be processed, with a default value of "train."

        Returns:
            The pipeline for training the network.
        """

        # assemble pipeline
        training_pipe = self.prenet_pipe(mode)

        if mode == "train":
            training_pipe += gp.PreCache(
                num_workers=self.num_workers, cache_size=self.cache_size
            )

        training_pipe += self.train_node

        for section in self.postnet_pipe(mode):
            if isinstance(section, list) or isinstance(section, tuple):
                for s in section:
                    training_pipe += s

            else:
                training_pipe += section

        if mode == "train" and self.snapshot_every is not None:
            snapshot_names: dict = {}
            if hasattr(self, "snapshot_arrays") and self.snapshot_arrays is not None:
                for array in self.snapshot_arrays:
                    snapshot_names[self.arrays[array]] = array
            else:
                for key, value in self.loss_input_dict.items():
                    if isinstance(key, str):
                        snapshot_names[value] = key
                    else:
                        snapshot_names[value] = value.identifier

            training_pipe += gp.Snapshot(
                dataset_names=snapshot_names,
                output_filename="{iteration}.zarr",
                every=self.snapshot_every,
            )  # add Snapshot node to save volumes

        # if mode == 'test':
        training_pipe += gp.PrintProfilingStats(self.save_every)

        return training_pipe

    def print_profiling_stats(self) -> None:
        """Prints the profiling statistics for the pipeline."""

        stats = "\n"
        stats += "Profiling Stats\n"
        stats += "===============\n"
        stats += "\n"
        stats += "NODE".ljust(20)
        stats += "METHOD".ljust(10)
        stats += "COUNTS".ljust(10)
        stats += "MIN".ljust(10)
        stats += "MAX".ljust(10)
        stats += "MEAN".ljust(10)
        stats += "MEDIAN".ljust(10)
        stats += "\n"

        summaries: list = list(self.batch.profiling_stats.get_timing_summaries().items())
        summaries.sort()

        for (node_name, method_name), summary in summaries:

            if summary.counts() > 0:
                stats += node_name[:19].ljust(20)
                stats += (
                    method_name[:19].ljust(10) if method_name is not None else " " * 10
                )
                stats += ("%d" % summary.counts())[:9].ljust(10)
                stats += ("%.2f" % summary.min())[:9].ljust(10)
                stats += ("%.2f" % summary.max())[:9].ljust(10)
                stats += ("%.2f" % summary.mean())[:9].ljust(10)
                stats += ("%.2f" % summary.median())[:9].ljust(10)
                stats += "\n"

        stats += "\n"

        print(stats)

    def train(self, iter: int) -> None:
        """Trains the model for the specified number of iterations.

        Args:
            iter (``integer``): 
                The number of iterations to train the model for.
        """

        self.model.train()
        training_pipeline = self.training_pipe()
        with gp.build(training_pipeline):
            pbar = trange(iter)
            for i in pbar:
                self.batch = training_pipeline.request_batch(self.batch_request)
                pbar.set_postfix({"loss": self.batch.loss})
                self.n_iter = self.train_node.iteration

                if i + 1 % self.log_every == 0:
                    self.train_node.summary_writer.flush()

    def test(self, mode: str = "train") -> gp.Batch:
        """Runs the testing mode for the model.

        Args:
            mode (str): The mode to run the test in.

        Returns:
            ``gp.Batch``:
                The test batch.
        """
        
        getattr(self.model, mode)()
        training_pipeline = self.training_pipe(mode="test")
        with gp.build(training_pipeline):
            self.batch: gp.Batch = training_pipeline.request_batch(self.batch_request)

        return self.batch
