import collections
import warnings

import numpy as np


class Metrics:
  """
  Lightweight metric aggregation utility used during training and evaluation.
  This class is heavily used in the training loop to accumulate metrics between logging intervals, reducing logging overhead.

  Attributes:
      _scalars (defaultdict[list]):
          Mapping from metric name to a list of scalar values collected since
          the last reset. Aggregated via mean in `result()`.
          - Stores lists of numeric values.
          - When `result()` is called, their mean value is returned.
          - Used for losses, rewards, costs, returns, etc.

      _lasts (dict):
          Mapping from metric name to the latest tensor-like value (e.g. images, videos, arrays). These overwrite previous entries.
          - Stores only the most recent value for non-scalar data.
          - Typically images, videos, or arrays that should not be averaged.
          - Passed directly to the logger unchanged.

  Methods:
      add(mapping, prefix=None):
          Add multiple metrics at once from a dictionary.
          Scalars are appended to `_scalars`, while arrays/tensors are stored
          as last values in `_lasts`. An optional prefix can be added to keys.

      result(reset=True):
          Returns a dictionary containing:
              - mean of all scalar metrics
              - latest tensor metrics
          Optionally clears internal buffers after aggregation.

      reset():
          Clears all stored metrics.
  """

  def __init__(self):
    self._scalars = collections.defaultdict(list)
    self._lasts = {}

  def scalar(self, key, value):
    self._scalars[key].append(value)

  def image(self, key, value):
    self._lasts[key].append(value)

  def video(self, key, value):
    self._lasts[key].append(value)

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        self._lasts[key] = value
      else:
        self._scalars[key].append(value)

  def result(self, reset=True):
    result = {}
    result.update(self._lasts)
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self._scalars.items():
        result[key] = np.nanmean(values, dtype=np.float64)
    reset and self.reset()
    return result

  def reset(self):
    self._scalars.clear()
    self._lasts.clear()
