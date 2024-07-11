## How to Train a CycleGAN

This is a basic outline of how to train a CycleGAN using the provided train script.

### Prerequisites

1. Python environment with necessary dependencies installed.
2. Image datasets for both source and target domains.
3. `raygun` repository cloned to your local machine.
4. Configuration file (train_conf.json) specifying training parameters.

### Configuration JSON Parameters
The configuration JSON file contains several parameters that you can modify to customize your CycleGAN training. Here are the key parameters:

- "framework": Specifies the deep learning framework to use (e.g. "torch", "tensorflow").
- "system": Specifies the type of system to use for training, so in this case  "CycleGAN".
- "job_command": Specifies the job command for running the training script (e.g "bsub", "-n 16", "-gpu "num=1"", "-q gpu_a100").
- "sources": Specifies the source domains and their corresponding paths, real names, and mask names.
- "common_voxel_size": Specifies the voxel size to cast all data into.
- "ndims": Specifies the number of dimensions for the input data.
- "batch_size": Specifies the batch size for training.
- "num_workers": Specifies the number of workers for data loading.
- "cache_size": Specifies the cache size for data loading.
- "scheduler": Specifies the scheduler type for adjusting learning rate during training.
- "scheduler_kwargs": Specifies the arguments for the scheduler.
- "g_optim_type": Specifies the optimizer type for the generator.
- "g_optim_kwargs": Specifies the arguments for the generator optimizer.
- "d_optim_type": Specifies the optimizer type for the discriminator.
- "d_optim_kwargs": Specifies the arguments for the discriminator optimizer.
- "loss_kwargs": Specifies the arguments for the loss functions.
- "gnet_type": Specifies the type of generator network architecture.
- "gnet_kwargs": Specifies the arguments for the generator network architecture.
- "pretrain_gnet": Specifies whether to pretrain the generator network.
- "dnet_type": Specifies the type of discriminator network architecture.
- "dnet_kwargs": Specifies the arguments for the discriminator network architecture.
- "spawn_subprocess": Specifies whether to spawn subprocesses for training.
- "side_length": Specifies the side length of the input image.
- "num_epochs": Specifies the number of training epochs.
- "log_every": Specifies the frequency of logging during training.
- "save_every": Specifies the frequency of saving models during training.
- "snapshot_every": Specifies the frequency of taking snapshots during training.

Here's an example of a CycleGan [configuration file]('../../experiments/ieee-isbi-2023/01_cycleGAN/train_conf.json)

### Training Methods

#### General training
- From the repository directory, run the following command:`rauygun-train CONFIG_FILE_LOCATION` where `CONGIF_FILE_LOCATION` is the relative path to the JSON configuration file for your training objective.

#### Batch training
- From the repository directory, run the following command: `rauygun-train-batch CONFIG_FILE_LOCATION` where `CONGIF_FILE_LOCATION` is the relative path to the JSON 

#### Cluster training
- From the repository directory, run the following command: `rauygun-train-cluster CONFIG_FILE_LOCATION` where `CONGIF_FILE_LOCATION` is the relative path to the JSON 


The CycleGAN training will start and progress will be displayed in the console.

Once the training is complete, the trained models will be saved in the specified output directory as per the configuration file.
