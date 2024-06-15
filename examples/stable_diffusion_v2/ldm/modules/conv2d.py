from mindspore import Parameter, nn, ops
from mindspore.common import initializer as init

class Conv2d(nn.Cell):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            pad_mode='same', 
            padding=0,
            dilation=1,
            group=1,
            has_bias=False,
            weight_init='normal',
            bias_init='zeros',
            data_format='NCHW'
        ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size

        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.has_bias = has_bias
        self.weight_init = weight_init
        self.bias_init = bias_init

        # Initialize weights and bias
        self.weight = Parameter(
            init.initializer(weight_init, [out_channels, in_channels // group, *self.kernel_size]), name="weight")
        if has_bias:
            self.bias = Parameter(init.initializer(bias_init, [out_channels]), name="bias")
        else:
            self.bias = None

    def construct(self, x):
        if self.pad_mode == 'same' or self.pad_mode == 'valid':
            return ops.extend.conv2d(x, self.weight, self.bias, self.stride, self.pad_mode, self.dilation, self.group)
        elif self.pad_mode == 'pad':
            if isinstance(self.padding, int):
                padding = (self.padding, self.padding, self.padding, self.padding)
                x = ops.pad_ext(x, padding)
                return ops.extend.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.group)
            else: #tuple/list
                x = ops.pad_ext(x, self.padding)
                return ops.extend.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.group)
        else:
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")
