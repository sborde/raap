"""This module contains utility function for working with HDF5 files."""

import h5py

import numpy as np

def list_keys(group) -> dict[str, str]:
    """Returns fieldname-key dictionary of the given HDF5 group."""
    if not isinstance(group, h5py.Group):
        raise TypeError("Expected an HDF5 group.")
    return {key.lower(): key for key in group.keys()}

def get_by_name(group: h5py.Group, name: str):
    """
    Get the HDF5 object from the given group 
    at the given name (case insensitive).
    """
    key = list_keys(group).get(name.lower(), None)
    if key is None:
        raise KeyError(f"Key '{name}' not found.")
    return group[key]

def get_group(group: h5py.Group, name: str) -> h5py.Group:
    """Get a group from an HDF5 file or group by name."""
    subgroup = get_by_name(group, name)
    if not isinstance(subgroup, h5py.Group):
        raise TypeError(f"Expected a group for '{name}'.")
    return subgroup

def get_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    """Get a dataset from an HDF5 group by name."""
    dataset = get_by_name(group, name)
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Expected a dataset for '{name}'.")
    return dataset

def get_ndarray(group: h5py.Group, name: str) -> np.ndarray:
    """Load a vector from an HDF5 group."""
    dataset = get_dataset(group, name)
    if not isinstance(dataset[:], np.ndarray):
        raise TypeError("Expected a numpy array.")
    return dataset[:]

def get_vector(group: h5py.Group, name: str) -> np.ndarray:
    """Load a vector from an HDF5 group."""
    return get_ndarray(group, name).squeeze()

def get_float(group: h5py.Group, name: str) -> float:
    """Load a scalar value from an HDF5 group."""
    dataset = get_dataset(group, name)
    return float(dataset[()].squeeze())

def get_int(group: h5py.Group, name: str) -> int:
    """Load an integer value from an HDF5 group."""
    return int(get_float(group, name))

def get_string(group: h5py.Group, name: str) -> str:
    """Load an encoded string from an HDF5 dataset."""
    string = get_dataset(group, name)[()]
    if isinstance(string, bytes):
        return string.decode('utf-8')
    if hasattr(string, 'astype'):
        return string.astype(str)
    return str(string)

def save_number(
    group: h5py.Group | h5py.File,
    name: str,
    value: float
):
    """Exports a scalar to an HDF5 dataset."""
    group.create_dataset(name, data=value)

def save_string(
    group: h5py.Group | h5py.File,
    name: str,
    value: str
):
    """Exports a string to an HDF5 dataset."""
    group.create_dataset(name, data=str(value).encode('utf-8'))

def save_array(
    group: h5py.Group | h5py.File,
    name: str,
    values: np.ndarray
):
    """Exports a numpy array to an HDF5 dataset."""
    group.create_dataset(
        name,
        data=values,
        compression="gzip",
        compression_opts=9
    )
