import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class LayerNorm(nn.extend.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def __init__(self, *args, **kwargs):
        if "eps" not in kwargs:
            kwargs["eps"] = 1e-5  # use 1e-5 epsilon by default same torch.nn.LayerNorm
        super().__init__(*args, **kwargs)

    def construct(self, x: Tensor):
        orig_type = x.dtype
        ret = super().construct(ops.cast(x, ms.float32))
        return ops.cast(ret, orig_type)


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)
