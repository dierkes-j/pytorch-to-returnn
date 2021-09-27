
from __future__ import annotations
from types import FunctionType
from typing import Dict, Any, Optional, List, Tuple, Union
import tensorflow as tf
from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init


class Stft(Module):
  def __init__(self, fft_size: int, frame_size: int, frame_shift: int, window: Optional[FunctionType] = None):
    super(Stft, self).__init__()
    self.fft_size = fft_size
    self.frame_size = frame_size
    self.frame_shift = frame_shift
    self.window = window

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    import ipdb
    ipdb.set_trace()
    return super(Stft, self)._get_output_shape_from_returnn(inputs_flat, layer)

  #def _get_output_shape_from_returnn(self,
  #                                   inputs_flat: List[Tensor], layer: LayerBase
  #                                   ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
  #  from ....naming import Naming
  #  naming = Naming.get_instance()
  #  x = naming.tensors[inputs_flat[0]]
  #  out_shape = (inputs_flat[0].shape[x.torch_axis_from_returnn_axis[x.returnn_data.batch_dim_axis]], 201, 1428)
  #  out_axis = {0:0, 1:1, 2:2}
  #  return out_shape, out_axis

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    existing_layer = {
      "class": "multichannel_stft_layer",
      "fft_size": self.fft_size,
      "frame_size": self.frame_size,
      "frame_shift": self.frame_shift,
      "window": "hanning",  # self.window,
      "from": [self._get_input_layer_name(input)]}
    eval_layer = {
      "class": "eval",
      "eval": self._eval_fn,
      "out_type": self._out_type,
      "eval_locals": {"fft_size": self.fft_size, "frame_size": self.frame_size, "frame_shift": self.frame_shift},
      "from": [self._get_input_layer_name(input)]}
    # fix size_placeholder?
    return eval_layer

  @staticmethod
  def _eval_fn(source, **kwargs):
    return tf.signal.stft(
      source(0, auto_convert=False), kwargs["frame_size"], kwargs["frame_shift"], kwargs["fft_size"])
    #return tf.transpose(tf.signal.stft(
    #  source(0, auto_convert=False), kwargs["frame_size"], kwargs["frame_shift"], kwargs["fft_size"]), perm=[0, 2, 1])


  @staticmethod
  def _out_type(sources, **kwargs):
    from returnn.tf.util.data import DimensionTag
    data = sources[0].output.copy_template()
    data = data.copy_add_feature_dim(2)
    #data._dim_tags[1].dimension = 1428
    data.dim = data._dim_tags[2].dimension = kwargs["fft_size"] // 2 + 1
    data.dtype = "complex64"
    return data


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]

