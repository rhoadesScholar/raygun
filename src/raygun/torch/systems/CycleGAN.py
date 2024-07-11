import itertools
import logging
from matplotlib import pyplot as plt
from raygun.io import CycleDataPipe
import torch
import daisy

import gunpowder as gp
import numpy as np

torch.backends.cudnn.benchmark = True

from raygun.torch.models import CycleModel
from raygun.torch.losses import LinkCycleLoss, SplitCycleLoss
from raygun.torch.optimizers import BaseDummyOptimizer, get_base_optimizer
from raygun.torch.systems import BaseSystem


logger: logging.Logger = logging.Logger(__name__, "INFO")

class CycleGAN(BaseSystem):
    """Implementation of a CycleGAN system for image-to-image translation using PyTorch.

    This class extends the `BaseSystem` class and implements the training and inference
    pipelines for a CycleGAN model.

    Args:
        config (``string``, optional):
            An optional path to CycleGAN configuration parameters for your system.
            These work to update/modified the default system parameters in the ``default_cycleGAN_conf.json``.
    """

    def __init__(self, config=None) -> None:
        super().__init__(
            default_config="../default_configs/default_cycleGAN_conf.json",
            config=config,
        )

        if self.common_voxel_size is None:
            self.common_voxel_size: gp.Coordinate = gp.Coordinate(
                daisy.open_ds(
                    self.sources["B"]["path"], self.sources["B"]["name"]
                ).voxel_size
            )
        else:
            self.common_voxel_size = gp.Coordinate(self.common_voxel_size)
        if self.ndims is None:
            self.ndims: int = sum(
                np.array(self.common_voxel_size) == np.min(self.common_voxel_size)
            )

    def batch_show(self, batch=None, i=0, show_mask=False) -> None:
        """Convenience method to display an output batch of a training process.

        Args:
            batch (gp.BatchRequest, optional): 
                Batch of training data to process. Defaults to None.

            i (``int``): 
                Index of the element in the batch to display. Defaults to 0.

            show_mask (``bool``): 
                Toggle to add a mask on the displayed batch. Defaults to False.
        """

        if batch is None:
            batch = self.batch
        if not hasattr(self, "col_dict"):
            self.col_dict = {"REAL": 0, "FAKE": 1, "CYCL": 2}
        if show_mask:
            self.col_dict["MASK"] = 3
        rows = (self.arrays["real_A"] in batch.arrays) + (
            self.arrays["real_B"] in batch.arrays
        )
        cols = 0
        for key in self.col_dict.keys():
            cols += key in [array.identifier[:4] for array in batch.arrays]
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        for array, value in batch.items():
            label = array.identifier
            if label[:4] in self.col_dict:
                c = self.col_dict[label[:4]]
                r = (int("_B" in label) + int("FAKE" in label)) % 2
                if len(value.data.shape) > 3:  # pick one from the batch
                    img = value.data[i].squeeze()
                else:
                    img = value.data.squeeze()
                if len(img.shape) == 3:
                    mid = img.shape[0] // 2  # for 3D volume
                    data = img[mid]
                else:
                    data = img

                if rows == 1:
                    ax = axes[c]
                elif cols == 1:
                    ax = axes[r]
                else:
                    ax = axes[r, c]

                ax.imshow(
                    data,
                    cmap="gray",
                    vmin=-int(data.min() < 0),
                    vmax=1 + 254 * int(data.max() > 1),
                )
                ax.set_title(label)

    def write_tBoard_graph(self, batch=None) -> None:
        """Writes the model graph to TensorBoard summary writer.
    
        Args:
            batch (gp.BatchRequest, optional): 
                Batch of training data. Defaults to None.
        """

        if batch is None:
            batch = self.trainer.batch

        ex_inputs = []
        for id in ["A", "B"]:
            array = self.datapipes[id].real
            if array in batch:
                ex_inputs += [torch.tensor(batch[array].data)]

        for i, ex_input in enumerate(ex_inputs):
            if self.ndims == len(
                self.common_voxel_size
            ):  # add channel dimension if necessary
                ex_input = ex_input.unsqueeze(axis=1)
            if self.batch_size == 1:  # ensure batch dimension is present
                ex_input = ex_input.unsqueeze(axis=0)
            ex_inputs[i] = ex_input

        try:
            self.trainer.train_node.summary_writer.add_graph(self.model, ex_inputs)
        except:
            self.logger.warning("Failed to add model graph to tensorboard.")

    def get_extents(self, side_length=None, array_name=None) -> gp.Coordinate:
        """Returns the extent of the data in the spatial dimensions.
    
        Args:
            side_length (``int``, optional): 
                Side length of the array. Defaults to None.
            array_name (``string``, optional): 
                Name of the array. Defaults to None.
            
        Returns:
            gp.Coordinate: 
                A tuple containing the extent of the data in the spatial dimensions.
        """
        if side_length is None:
            side_length = self.side_length

        if ("padding_type" in self.gnet_kwargs) and (
            self.gnet_kwargs["padding_type"].lower() == "valid"
        ):
            if array_name is not None and not (
                "real" in array_name.lower() or "mask" in array_name.lower()
            ):
                shape = (1, 1) + (side_length,) * self.ndims
                pars = [par for par in self.netG1.parameters()]
                result = self.netG1(torch.zeros(*shape, device=pars[0].device))
                if "fake" in array_name.lower():
                    side_length = result.shape[-1]
                elif "cycle" in array_name.lower():
                    result = self.netG1(result)
                    side_length = result.shape[-1]

        extents = np.ones((len(self.common_voxel_size)))
        extents[
            -self.ndims :
        ] = side_length  # assumes first dimension is z (i.e. the dimension breaking isotropy)
        return gp.Coordinate(extents)

    def setup_networks(self) -> None:
        self.netG1 = self.get_network(self.gnet_type, self.gnet_kwargs)
        self.netG2 = self.get_network(self.gnet_type, self.gnet_kwargs)

        self.netD1 = self.get_network(self.dnet_type, self.dnet_kwargs)
        self.netD2 = self.get_network(self.dnet_type, self.dnet_kwargs)

    def setup_model(self) -> None:
        if not hasattr(self, "netG1"):
            self.setup_networks()

        if self.sampling_bottleneck:
            scale_factor_A: tuple = tuple(
                np.divide(self.common_voxel_size, self.A_voxel_size)[-self.ndims :]
            )
            if not any([s < 1 for s in scale_factor_A]):
                scale_factor_A = None
            scale_factor_B: tuple = tuple(
                np.divide(self.common_voxel_size, self.B_voxel_size)[-self.ndims :]
            )
            if not any([s < 1 for s in scale_factor_B]):
                scale_factor_B = None
        else:
            scale_factor_A, scale_factor_B = None, None

        self.model:CycleModel = CycleModel(
            self.netG1,
            self.netG2,
            scale_factor_A,
            scale_factor_B,
            split=self.loss_type.lower() == "split",
            freeze_norms_at=self.freeze_norms_at,
        )

    def setup_optimization(self) -> None:
        self.optimizer_D = get_base_optimizer(self.d_optim_type)(
            itertools.chain(self.netD1.parameters(), self.netD2.parameters()),
            **self.d_optim_kwargs
        )

        if hasattr(self, "scheduler"):
            scheduler = self.scheduler
            scheduler_kwargs = self.scheduler_kwargs
        else:
            scheduler = None
            scheduler_kwargs:dict = {}

        if self.loss_type.lower() == "link":
            self.optimizer_G = get_base_optimizer(self.g_optim_type)(
                itertools.chain(self.netG1.parameters(), self.netG2.parameters()),
                **self.g_optim_kwargs
            )
            self.optimizer: BaseDummyOptimizer = BaseDummyOptimizer(
                optimizer_G=self.optimizer_G,
                optimizer_D=self.optimizer_D,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
            )

            self.loss:LinkCycleLoss = LinkCycleLoss(
                self.netD1,
                self.netG1,
                self.netD2,
                self.netG2,
                self.optimizer_G,
                self.optimizer_D,
                self.ndims,
                **self.loss_kwargs
            )

        elif self.loss_type.lower() == "split":
            self.optimizer_G1 = get_base_optimizer(self.g_optim_type)(
                self.netG1.parameters(), **self.g_optim_kwargs
            )
            self.optimizer_G2 = get_base_optimizer(self.g_optim_type)(
                self.netG2.parameters(), **self.g_optim_kwargs
            )
            self.optimizer = BaseDummyOptimizer(
                optimizer_G1=self.optimizer_G1,
                optimizer_G2=self.optimizer_G2,
                optimizer_D=self.optimizer_D,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
            )

            self.loss = SplitCycleLoss(
                self.netD1,
                self.netG1,
                self.netD2,
                self.netG2,
                self.optimizer_G1,
                self.optimizer_G2,
                self.optimizer_D,
                self.ndims,
                **self.loss_kwargs
            )

        else:
            raise NotImplementedError(
                "Unexpected Loss Style. Accepted options are 'cycle' or 'split'"
            )

    def setup_datapipes(self) -> None:
        self.arrays:dict = {}
        self.datapipes:dict = {}
        for id, src in self.sources.items():
            self.datapipes[id] = CycleDataPipe(
                id,
                src,
                self.ndims,
                self.common_voxel_size,
                self.interp_order,
                self.batch_size,
            )
            self.arrays.update(self.datapipes[id].arrays)

    def make_request(self, mode: str = "train") -> gp.BatchRequest:
        """Creates a BatchRequest object for the specified mode.

        Args:
            mode (``string``): 
                The processing mode to create the BatchRequest for. Can be one of "train", "val", "test", or "predict".
            
        Returns:
            A BatchRequest object specifying the desired output arrays and their requested extents and voxel sizes.
        """
        # create request
        request:gp.BatchRequest = gp.BatchRequest()
        for array_name, array in self.arrays.items():
            if (
                mode == "prenet" and ("real" in array_name or "mask" in array_name)
            ) or (
                mode != "prenet" and (mode != "predict" or "cycle" not in array_name)
            ):
                extents: gp.Coordinate = self.get_extents(array_name=array.identifier)
                request.add(
                    array, self.common_voxel_size * extents, self.common_voxel_size
                )
        return request


if __name__ == "__main__":
    system: CycleGAN = CycleGAN(config="./train_conf.json")
    system.logger.info("CycleGAN system loaded. Training...")
    _: None = system.train()
    system.logger.info("Done training!")
