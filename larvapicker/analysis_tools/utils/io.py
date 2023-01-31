from typing import Tuple, Union, List

import h5py
import numpy as np
import json

from pathlib import Path


class TimestampedArrayWriter:
	def __init__(
			self, src,
			filename: str,
			shape: Tuple[int, ...],
			dtype: np.dtype,
			groupname: Union[None, str] = None,
			compression="lzf",
			compression_opts=None):
		""" src must yield numpy arrays with shape and dtype matching the shape
		and dtype provided."""

		self.src = src

		self.shape = shape
		self.dtype = dtype

		self.N_complete = 0

		self.filename = filename
		self.file = h5py.File(filename, "w")

		if groupname is None:
			groupname = "/"
			self.group = self.file["/"]
		else:
			if groupname[0] != "/":
				groupname = "/" + groupname
			self.group = self.file.create_group(groupname)

		self.data = self.group.create_dataset(
			"data", (0, *shape),
			chunks=(1, *shape),
			dtype=dtype,
			compression=compression,
			compression_opts=compression_opts,
			maxshape=(None, *shape)
		)

		self.times = self.group.create_dataset(
			"times", (0,),
			chunks=(1,),
			dtype=np.dtype("float64"),
			maxshape=(None,)
		)

	@classmethod
	def from_source(cls, src, filename: str, groupname: Union[None, str] = None):
		"""If the source has shape and dtype fields, this can be used to
		construct the writer more succinctly."""
		return cls(src, filename, src.shape, src.dtype, groupname)

	def close(self):
		self.file.close()

	def append_data(self, msg):
		(t, x) = msg

		self.N_complete += 1

		self.data.resize((self.N_complete, *self.shape))
		self.times.resize((self.N_complete,))

		self.data[self.N_complete - 1, ...] = x
		self.times[self.N_complete - 1] = t


def save_data(src,
			  filename: str,
			  tlist=None,
			  compression="lzf",
			  compression_opts=None):
	file = h5py.File(filename, "w")
	data = file.create_dataset(
		"data", src.shape,
		dtype=src.dtype,
		compression=compression,
		compression_opts=compression_opts,
	)
	data[...] = src
	times = file.create_dataset(
		"times", (src.shape[0],),
		dtype=np.float64
	)
	times[...] = np.arange(src.shape[0]) if tlist is None else tlist
	file.close()


def get_metadata(dataset_path: Path):
	json_filename = dataset_path / "metadata.json"
	with open(json_filename) as json_file:
		metadata = json.load(json_file)
		return metadata


def get_times(dataset_path: Path, file_name=None) -> np.ndarray:
	"""Return the timestamp of a given data frame."""
	if file_name is not None:
		f = h5py.File(dataset_path / file_name, 'r')
	else:
		f = h5py.File(dataset_path / "data.h5", 'r')

	return f["times"][:]


def get_slice(dataset_path: Path, t: int, file_name=None) -> np.ndarray:
	"""Return a slice from the "data" field of an HDF5 file.
	Time (t) is assumed to be the last (longest-stride) dimension. This should
	generally be cached to avoid opening and closing files too many times:

		my_get_slice = lru_cache(get_h5_slice, maxsize=1000)

	"""
	if file_name is not None:
		f = h5py.File(dataset_path / file_name, 'r')
	else:
		f = h5py.File(dataset_path / "data.h5", 'r')

	return f["data"][t]


def get_slices(dataset_path: Path, times: List[int], file_name=None) -> np.ndarray:
	"""Return volumes of the specified time points in the dataset."""

	if file_name is not None:
		f = h5py.File(dataset_path / file_name, 'r')
	else:
		f = h5py.File(dataset_path / "data.h5", 'r')

	return f["data"][times]


def apply_lut(x: np.ndarray, lo: float, hi: float, newtype=None) -> np.ndarray:
	"""Clip x to the range [lo, hi], then rescale to fill the range of
	newtype."""

	if newtype is None:
		newtype = x.dtype

	y_float = (x-lo)/(hi-lo)
	y_clipped = np.clip(y_float, 0, 1)

	if np.issubdtype(newtype, np.integer):
		maxval = np.iinfo(newtype).max
	else:
		maxval = 1.0

	return (maxval*y_clipped).astype(newtype)


def auto_lut(x: np.ndarray, quantiles=(0.5,0.99), newtype=None) -> np.ndarray:
	"""Linearly map the specified quantiles of x to the range of newtype."""

	lo = np.quantile(x, quantiles[0])
	hi = np.quantile(x, quantiles[1])

	return apply_lut(x, lo, hi, newtype=newtype)


def compare_red_green(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Autoscale 2D arrays x and y, then stack them with a blank blue channel
	to form a 3D array with shape (3, size_Y, size_X) and type uint8."""

	r = auto_lut(x, newtype=np.uint8)
	g = auto_lut(y, newtype=np.uint8)
	b = np.zeros_like(r)

	return np.dstack([r, g, b])
