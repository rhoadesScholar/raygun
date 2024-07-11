from abc import abstractmethod
import functools
from glob import glob
import re
import logging
import os
import random

import numpy as np
import torch

from raygun.torch.utils import read_config
from raygun.torch import networks
from raygun.torch.networks.utils import init_weights
from raygun.torch import train

parent_dir = os.path.dirname(os.path.dirname(__file__))


class BaseSystem:
    def __init__(
        self, default_config="../default_configs/blank_conf.json", config=None
    ) -> None:
        # Add default params
        default_config = default_config.replace("..", parent_dir)
        for key, value in read_config(default_config).items():
            setattr(self, key, value)

        if config is not None:
            # Get this configuration
            for key, value in read_config(config).items():
                setattr(self, key, value)

        if not hasattr(self, "checkpoint_basename"):
            try:
                self.checkpoint_basename = os.path.join(
                    self.model_path, self.model_name
                )
            except:
                self.checkpoint_basename = "./models/model"

        if not hasattr(self, "checkpoint") or self.checkpoint is None:
            try:
                self.checkpoint, self.iteration = self._get_latest_checkpoint()
            except:
                print("Checkpoint not found. Starting from scratch.")
                self.checkpoint = None

        if hasattr(self, "random_seed") and self.random_seed is not None:
            self.set_random_seed()

    def batch_show(self):
        """Implement in subclasses."""
        raise NotImplementedError()

    def arrays_min_max(
        self,
        batch=None,
        lims={bool: [True, True], np.float32: [0, 1]},
        test=True,
        show=False,
    ):
        if batch is None:
            if hasattr(self, "batch"):
                batch = self.batch
            else:
                print("No batch arrays available.")
                return

        for name, array in batch.arrays.items():
            if show:
                print(f"{name}: min={array.data.min()}  <--> max={array.data.max()}")

            if test and array.data.dtype in lims.keys():
                assert array.data.min() >= lims[array.data.dtype][0]
                assert array.data.max() <= lims[array.data.dtype][1]

    def set_random_seed(self):
        if self.random_seed is None:
            self.random_seed = 42
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def set_verbose(self, verbose=None):
        if verbose is not None:
            self.verbose = verbose
        elif self.verbose is None:
            self.verbose = True
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def set_device(self, id=0):
        self.device_id = id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
        # torch.cuda.set_device(id) # breaks spawning subprocesses

    def load_saved_model(self, checkpoint=None, cuda_available=None):
        if not hasattr(self, "model"):
            self.setup_model()

        if cuda_available is None:
            cuda_available = torch.cuda.is_available()

        if checkpoint is None:
            checkpoint = self.checkpoint
        else:
            self.checkpoint = checkpoint

        if checkpoint is not None:
            if not cuda_available:
                checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
            else:
                checkpoint = torch.load(checkpoint)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.logger.warning("No saved checkpoint found.")

    def _get_latest_checkpoint(self):
        basename = self.model_path + self.model_name

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r"(\d+)", text)]

        checkpoints = glob(basename + "_checkpoint_*")
        checkpoints.sort(key=natural_keys)

        if len(checkpoints) > 0:

            checkpoint = checkpoints[-1]
            iteration = int(checkpoint.split("_")[-1])
            return checkpoint, iteration

        return None, 0

    def get_downsample_factors(self, net_kwargs):
        if "downsample_factors" not in net_kwargs:
            down_factor = (
                2 if "down_factor" not in net_kwargs else net_kwargs.pop("down_factor")
            )
            num_downs = (
                3 if "num_downs" not in net_kwargs else net_kwargs.pop("num_downs")
            )
            net_kwargs.update(
                {
                    "downsample_factors": [
                        (down_factor,) * self.ndims,
                    ]
                    * (num_downs - 1)
                }
            )
        return net_kwargs

    def get_network(self, net_type="unet", net_kwargs=None):
        # TODO: make general call of: net = getattr(networks, net_type)(**net_kwargs)
        if "final_activation" in net_kwargs.keys():
            final_activation = net_kwargs.pop("final_activation")
        else:
            final_activation = None

        if "output_nc" in net_kwargs.keys():
            output_nc = net_kwargs.pop("output_nc")
        else:
            output_nc = net_kwargs["input_nc"]

        if isinstance(final_activation, str):
            final_activation = getattr(torch.nn, final_activation)

        add_final = True
        if net_type == "unet":
            net_kwargs = self.get_downsample_factors(net_kwargs)

            net = networks.UNet(**net_kwargs)

        elif net_type == "residualunet":
            net_kwargs = self.get_downsample_factors(net_kwargs)

            net = networks.ResidualUNet(**net_kwargs)

        elif net_type == "resnet":
            net = networks.ResNet(self.ndims, **net_kwargs)

        elif net_type == "patchdiscriminator":
            norm_instance = {
                2: torch.nn.InstanceNorm2d,
                3: torch.nn.InstanceNorm3d,
            }[self.ndims]
            net_kwargs["norm_layer"] = functools.partial(
                norm_instance, affine=False, track_running_stats=False
            )
            net = networks.NLayerDiscriminator(self.ndims, **net_kwargs)
            add_final = False

        elif hasattr(networks, net_type):
            net = getattr(networks, net_type)(**net_kwargs)

        else:
            raise f"Unknown discriminator type requested: {net_type}"

        if add_final:
            layers = [
                net,
                getattr(torch.nn, f"Conv{self.ndims}d")(
                    net_kwargs["ngf"],
                    output_nc,
                    (1,) * self.ndims,
                    padding="valid"
                    if "padding_type" not in net_kwargs.keys()
                    else net_kwargs["padding_type"],
                ),
            ]
            if final_activation is not None:
                layers.append(final_activation)

            net = torch.nn.Sequential(*layers)

        activation = (
            net_kwargs["activation"] if "activation" in net_kwargs else torch.nn.ReLU
        )
        if activation is not None:
            init_weights(
                net,
                init_type="kaiming",
                nonlinearity=activation.__class__.__name__.lower(),
            )
        elif net_type == "classic":
            init_weights(net, init_type="kaiming")
        else:
            init_weights(
                net, init_type="normal", init_gain=0.05
            )  # TODO: MAY WANT TO ADD TO CONFIG FILE

        return net

    def get_valid_context(self, net_kwargs, side_length=None):
        # returns number of pixels to crop from a side to trim network outputs to valid FOV
        if side_length is None:
            side_length = self.side_length

        net_kwargs["padding_type"] = "valid"
        net = self.get_network(gnet_kwargs=net_kwargs)

        shape = (1, 1) + (side_length,) * self.ndims
        pars = [par for par in net.parameters()]
        result = net(torch.zeros(*shape, device=pars[0].device))
        return np.ceil((np.array(shape) - np.array(result.shape)) / 2)[-self.ndims :]

    @abstractmethod
    def setup_networks(self):
        """Implement in subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def setup_model(self):
        """Implement in subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def setup_optimization(self):
        """Implement in subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def setup_datapipes(self):
        """Implement in subclasses."""
        raise NotImplementedError()

    @abstractmethod
    def make_request(self, mode):
        """Implement in subclasses."""
        raise NotImplementedError()

    def setup_trainer(self):
        trainer_base = getattr(train, self.trainer_base)
        if hasattr(self, "train_kwargs"):
            self.trainer = trainer_base(
                self.datapipes,
                self.make_request(mode="train"),
                self.model,
                self.loss,
                self.optimizer,
                **self.train_kwargs,
            )
        else:  # backwards compatability: Remove in 0.3.0
            self.trainer = trainer_base(
                self.datapipes,
                self.make_request(mode="train"),
                self.model,
                self.loss,
                self.optimizer,
                self.tensorboard_path,
                self.log_every,
                self.checkpoint_basename,
                self.save_every,
                self.spawn_subprocess,
                self.num_workers,
                self.cache_size,
                snapshot_every=self.snapshot_every,
            )
        
        self.arrays.update(self.trainer.arrays)

    def build_system(self) -> None:
        # define our network model for training
        self.setup_networks()
        self.setup_model()
        self.setup_optimization()
        self.setup_datapipes()
        self.setup_trainer()

    def train(self) -> None:
        if not hasattr(self, "trainer"):
            self.build_system()
        if hasattr(self, "train_kwargs"):
            self.trainer.train(self.train_kwargs["num_epochs"])
        else:  # backwards compatability -> TODO: remove in 0.3.0
            self.trainer.train(self.num_epochs)

    def test(self, mode: str = "train"):  # set to 'train' or 'eval'
        if not hasattr(self, "trainer"):
            self.build_system()
        self.batch = self.trainer.test(mode)
        try:
            self.batch_show()
        except:
            pass  # if not implemented
        return self.batch
