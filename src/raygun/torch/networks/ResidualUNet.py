# adapted from funlib.learn.torch.models

from funlib.learn.torch.models.conv4d import Conv4d
import math
import numpy as np
import torch
import torch.nn as nn

from raygun.torch.networks.utils import NoiseBlock, ParameterizedNoiseBlock


class ConvPass(torch.nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        kernel_sizes,
        activation,
        padding="valid",
        residual=False,
        padding_mode="reflect",
        norm_layer=None,
        final=False,
    ):
        """Convolution pass block

        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            kernel_sizes (list(int) or array_like): Kernel sizes for convolution layers.
            activation (str or callable): Name of activation function in 'torch.nn' or the function itself.
            padding (str, optional): What type of padding to use in convolutions. Defaults to 'valid'.
            residual (bool, optional): Whether to make the blocks calculate the residual. Defaults to False.
            padding_mode (str, optional): What values to use in padding (i.e. 'zeros', 'reflect', 'wrap', etc.). Defaults to 'reflect'.
            norm_layer (callable or None, optional): Whether to use a normalization layer and if so (i.e. if not None), the layer to use. Defaults to None.
            final (bool, optional): Whether this block is the final output of the network (and thus should have the final activation omitted). Defaults to False.

        Returns:
            ConvPass: Convolution block
        """
        super(ConvPass, self).__init__()

        if activation is not None:
            if isinstance(activation, str):
                self.activation = getattr(torch.nn, activation)()
            else:
                self.activation = activation()  # assume is function
        else:
            self.activation = nn.Identity()

        self.residual = residual
        self.padding = padding
        self.final = final

        layers = []

        for i, kernel_size in enumerate(kernel_sizes):

            self.dims = len(kernel_size)

            conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d, 4: Conv4d}[self.dims]

            try:
                layers.append(
                    conv(
                        input_nc,
                        output_nc,
                        kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )
                if residual and i == 0:
                    if input_nc < output_nc:
                        groups = input_nc
                    else:
                        groups = output_nc
                    self.x_init_map = conv(
                        input_nc,
                        output_nc,
                        np.ones(self.dims, dtype=int),
                        padding=padding,
                        padding_mode=padding_mode,
                        bias=False,
                        groups=groups,
                    )
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            if norm_layer is not None:
                layers.append(norm_layer(output_nc))

            if not (
                (residual and i == (len(kernel_sizes) - 1))
                or (final and i == (len(kernel_sizes) - 1))
            ):
                layers.append(self.activation)

            input_nc = output_nc

        self.conv_pass = torch.nn.Sequential(*layers)

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        if not self.residual:
            return self.conv_pass(x)
        else:
            res = self.conv_pass(x)
            if self.padding.lower() == "valid":
                init_x = self.crop(self.x_init_map(x), res.size()[-self.dims :])
            else:
                init_x = self.x_init_map(x)
            if not self.final:
                return self.activation(init_x + res)
            else:
                return init_x + res


class ConvDownsample(torch.nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        kernel_sizes,
        downsample_factor,
        activation,
        padding="valid",
        padding_mode="reflect",
        norm_layer=None,
    ):
        """Convolution-based downsampling

        Args:
            input_nc (int): Number of input channels.
            output_nc (int): Number of output channels.
            kernel_sizes (list(int) or array_like): Kernel sizes for convolution layers.
            downsample_factor (int): Factor by which to downsample in all spatial dimensions.
            activation (str or callable): Name of activation function in 'torch.nn' or the function itself.
            padding (str, optional): What type of padding to use in convolutions. Defaults to 'valid'.
            padding_mode (str, optional): What values to use in padding (i.e. 'zeros', 'reflect', 'wrap', etc.). Defaults to 'reflect'.
            norm_layer (callable or None, optional): Whether to use a normalization layer and if so (i.e. if not None), the layer to use. Defaults to None.

        Returns:
            Downsampling layer.
        """

        super(ConvDownsample, self).__init__()

        if activation is not None:
            if isinstance(activation, str):
                self.activation = getattr(torch.nn, activation)()
            else:
                self.activation = activation()  # assume is function
        else:
            self.activation = nn.Identity()

        self.padding = padding

        layers = []

        self.dims = len(kernel_sizes)

        conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d, 4: Conv4d}[self.dims]

        try:
            layers.append(
                conv(
                    input_nc,
                    output_nc,
                    kernel_sizes,
                    stride=downsample_factor,
                    # padding=padding, #TODO: Make work with same padding
                    padding="valid",
                    padding_mode=padding_mode,
                )
            )

        except KeyError:
            raise RuntimeError("%dD convolution not implemented" % self.dims)

        if norm_layer is not None:
            layers.append(norm_layer(output_nc))

        layers.append(self.activation)
        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_pass(x)


class MaxDownsample(torch.nn.Module):
    def __init__(self, downsample_factor, flexible=True):
        """MaxPooling-based downsampling

        Args:
            downsample_factor (list(int) or array_like): Factors to downsample by in each dimension.
            flexible (bool, optional): True allows torch.nn.MaxPoolNd to crop the right/bottom of tensors in order to allow pooling of tensors not evenly divisible by the downsample_factor. Alternative implementations could pass 'ceil_mode=True' or 'padding= {# > 0}' to avoid cropping of inputs. False forces inputs to be evenly divisible by the downsample_factor, which generally restricts the flexibility of model architectures. Defaults to True.

        Returns:
            Downsampling layer.
        """

        super(MaxDownsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor
        self.flexible = flexible

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d,  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(
            downsample_factor,
            stride=downsample_factor,
        )

    def forward(self, x):
        if self.flexible:
            try:
                return self.down(x)
            except:
                self.check_mismatch(x.size())
        else:
            self.check_mismatch(x.size())
            return self.down(x)

    def check_mismatch(self, size):
        for d in range(1, self.dims + 1):
            if size[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (size, self.downsample_factor, self.dims - d)
                )
        return


class Upsample(torch.nn.Module):
    def __init__(
        self,
        scale_factor,
        mode=None,
        input_nc=None,
        output_nc=None,
        crop_factor=None,
        next_conv_kernel_sizes=None,
    ):

        super(Upsample, self).__init__()

        if crop_factor is not None:
            assert (
                next_conv_kernel_sizes is not None
            ), "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes
        self.dims = len(scale_factor)

        if mode == "transposed_conv":

            up = {2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}[self.dims]

            self.up = up(
                input_nc, output_nc, kernel_size=scale_factor, stride=scale_factor
            )

        else:

            self.up = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

    def crop_to_factor(self, x, factor, kernel_sizes):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        """

        shape = x.size()
        spatial_shape = shape[-self.dims :]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes) for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n * f + c for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:

            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with factor %s and following "
                "convolutions %s" % (shape, factor, kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):

        g_up = self.up(g_out)

        if self.crop_factor is not None:
            g_cropped = self.crop_to_factor(
                g_up, self.crop_factor, self.next_conv_kernel_sizes
            )
        else:
            g_cropped = g_up

        f_cropped = self.crop(f_left, g_cropped.size()[-self.dims :])

        return torch.cat([f_cropped, g_cropped], dim=1)


class ResidualUNet(torch.nn.Module):
    def __init__(
        self,
        input_nc,
        ngf,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        activation="ReLU",
        output_nc=None,
        num_heads=1,
        constant_upsample=False,
        downsample_method="max",
        padding_type="valid",
        residual=False,
        norm_layer=None,
        add_noise=False,
    ) -> None:
        """Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        It will perform 4D convolutions as long as ``length`` is greater than 1.
        As soon as ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            input_nc:

                The number of input channels.

            ngf:

                The number of feature maps in the first layer. By default, this is also the
                number of output feature maps. Stored in the ``channels``
                dimension.

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            output_nc (optional):

                The number of feature maps in the output layer. By default, this is the same as the
                number of feature maps of the input layer. Stored in the ``channels``
                dimension.

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            downsample_method (optional):

                Whether to use max pooling ('max') or strided convolution ('convolve') for downsampling layers. Default is max pooling.

            padding_type (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).

            residual (optional):

                Whether to train convolutional layers to output residuals to add to inputs (as in ResNet) or to directly convolve input data to output. Either 'True' or 'False' (default).

            norm_layer (optional):

                What, if any, normalization to layer after network layers. Default is none.

            add_noise (optional):

                Whether to add gaussian noise with 0 mean and unit variance ('True'), mean and variance parameterized by the network ('param'), or no noise ('False' <- default).

        """

        super(ResidualUNet, self).__init__()

        self.ndims = len(downsample_factors[0])
        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc else ngf
        self.residual = residual
        self.padding_type = padding_type

        if activation is not None:
            if isinstance(activation, str):
                self.activation = getattr(torch.nn, activation)()
            else:
                self.activation = activation()  # assume is function
        else:
            self.activation = nn.Identity()

        if add_noise == "param":  # add noise feature if necessary
            self.noise_layer = ParameterizedNoiseBlock()
        elif add_noise:
            self.noise_layer = NoiseBlock()
        else:
            self.noise_layer = None
        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [
                [(3,) * self.ndims, (3,) * self.ndims]
            ] * self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3,) * self.ndims, (3,) * self.ndims]] * (
                self.num_levels - 1
            )

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if padding_type.lower() == "valid":
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f * ff for f, ff in zip(factor, factor_product)
                    )
            elif padding_type.lower() == "same":
                factor_product = None
            else:
                raise f"Invalid padding_type option: {padding_type}"
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList(
            [
                ConvPass(
                    input_nc
                    if level == 0
                    else ngf
                    * fmap_inc_factor ** (level - (downsample_method.lower() == "max")),
                    ngf * fmap_inc_factor**level,
                    kernel_size_down[level],
                    activation=activation,
                    padding=padding_type,
                    residual=self.residual,
                    norm_layer=norm_layer,
                )
                for level in range(self.num_levels)
            ]
        )
        self.dims = self.l_conv[0].dims

        # left downsample layers
        if downsample_method.lower() == "max":

            self.l_down = nn.ModuleList(
                [
                    MaxDownsample(downsample_factors[level])
                    for level in range(self.num_levels - 1)
                ]
            )

        elif downsample_method.lower() == "convolve":

            self.l_down = nn.ModuleList(
                [
                    ConvDownsample(
                        ngf * fmap_inc_factor**level,
                        ngf * fmap_inc_factor ** (level + 1),
                        kernel_size_down[level][0],
                        downsample_factors[level],
                        activation=activation,
                        padding=padding_type,
                        norm_layer=norm_layer,
                    )
                    for level in range(self.num_levels - 1)
                ]
            )

        else:

            raise RuntimeError(
                f'Unknown downsampling method {downsample_method}. Use "max" or "convolve" instead.'
            )

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Upsample(
                            downsample_factors[level],
                            mode="nearest" if constant_upsample else "transposed_conv",
                            input_nc=ngf * fmap_inc_factor ** (level + 1)
                            + (level == 1 and (add_noise is not False)),
                            output_nc=ngf * fmap_inc_factor ** (level + 1),
                            crop_factor=crop_factors[level],
                            next_conv_kernel_sizes=kernel_size_up[level],
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

        # right convolutional passes
        self.r_conv = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvPass(
                            ngf * fmap_inc_factor**level
                            + ngf * fmap_inc_factor ** (level + 1),
                            ngf * fmap_inc_factor**level
                            if output_nc is None or level != 0
                            else output_nc,
                            kernel_size_up[level],
                            activation=activation,
                            padding=padding_type,
                            residual=self.residual,
                            norm_layer=norm_layer,
                            final=(level == 0),
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:

            if self.noise_layer is not None:
                f_left = self.noise_layer(f_left)
            fs_out = [f_left] * self.num_heads

        else:

            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            # up, concat, and crop
            fs_right = [
                self.r_up[h][i](f_left, gs_out[h]) for h in range(self.num_heads)
            ]

            # convolve
            fs_out = [self.r_conv[h][i](fs_right[h]) for h in range(self.num_heads)]

        return fs_out

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.ndims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):

        y = self.rec_forward(self.num_levels - 1, x)
        if self.padding_type.lower() == "valid":
            x = self.crop(x, y[0].size()[-self.ndims :])

        for i in range(self.num_heads):
            y[i] = self.activation(x + y[i])

        if self.num_heads == 1:
            return y[0]

        return y
