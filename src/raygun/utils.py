import inspect
from io import StringIO
from jsmin import jsmin
import json
import os
import numpy as np
import gunpowder as gp


def passing_locals(local_dict:dict) -> dict:
    """Extracts the local variables from a given dictionary and returns them as a keyword argument dictionary.

        Args:
        local_dict (``dict``): 
            A dictionary containing local variables.

        Returns:
        ``dict``: 
            A dictionary containing the extracted keyword arguments.
    """

    kwargs:dict = {}
    for k, v in local_dict.items():
        if k[0] != "_" and k != "self":
            if k == "kwargs":
                kwargs.update(v)
            else:
                kwargs[k] = v
    return kwargs


def get_config_name(config_path:str, base_folder:str) -> str:
    """Returns a string containing the configuration name given a configuration file path and a base folder.

        Args:
            config_path (``string``): 
                Path of the configuration file.
            base_folder (``string``): 
                Base folder of the configuration file.

        Returns:
            ``string``: 
                Configuration name.
    """

    config_name: str = os.path.dirname(config_path)
    config_name: str = config_name.replace(base_folder, "")
    config_name: str = "_".join(config_name.split("/"))[1:]

    return config_name


def calc_max_padding(
    output_size, voxel_size, neighborhood=None, sigma=None, mode="shrink"
):
    """Calculate the maximum padding for an output size given the voxel size and optional parameters.

    Args:
        output_size (Tuple[int, int, int]): 
            The size of the output.

        voxel_size (Tuple[float, float, float]): 
            The size of the voxels.
            
        neighborhood (Tuple[Tuple[int, int, int], ...]], optional): 
            A tuple of 3x3x3 neighborhood values.

        sigma (``float``, optional): 
            The sigma value for Gaussian padding.

        mode (``string``, optional): 
            The mode for snapping the output to the grid.

    Returns:
        Tuple[int, int, int]: 
            The maximum padding for the output size.
    """

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity: gp.Coordinate = gp.Coordinate(
            [np.abs(aff) for val in neighborhood for aff in val if aff != 0]
        )

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding: gp.Coordinate = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding: gp.Roi = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


def serialize(obj):
    """Serialize a Python object into a JSON-compatible format.

    Args:
        obj (``dict``, ``np.ndarray``, ``np.int64``, ``obj``, other): 
            A Python object to be serialized.

    Returns:
        (``integer``, ``string``, ``dict``)
            The serialized object in a JSON-compatible format.
    """

    if isinstance(obj, dict):
        out: dict = {}
        for key, value in obj.items():
            out[key] = serialize(value)
        return out
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif inspect.isclass(obj):
        return f"#{'.'.join([obj.__module__, obj.__name__])}#"
    else:
        try:
            json.dumps(obj)
            return obj
        except:
            return f"#{repr(obj)}#"


def to_json(obj, file:str, indent=3) -> None:
    """Serializes the given object to JSON and writes it to a file.

    Args:
        obj (``dict``, ``np.ndarray``, ``np.int64``, ``obj``, other): 
            The object to serialize to JSON.

        file (``string``): 
            The name of the file to write the serialized JSON to.

        indent (``integer``): 
            The number of spaces to use for indentation in the JSON file.
    """

    out = serialize(obj)
    with open(file, "w") as f:
        json.dump(out, f, indent=indent)


def load_json_file(fin:str) -> dict:
    """Loads a JSON file from disk and parses it into a Python dictionary.

    Args:
        fin (``string``): 
            The name of the file to load the JSON data from.

    Returns:
        ``dict``:
            A Python dictionary containing the parsed JSON data.
    """

    with open(fin, "r") as f:
        config = json.load(StringIO(jsmin(f.read())))
    return config


def merge_dicts(from_dict:dict, to_dict:dict) -> dict:
    """Merges two dictionaries together, with keys in the `from_dict` dictionary taking
    precedence over keys in the `to_dict` dictionary.

    Args:
        from_dict (``dict``): 
            The dictionary to merge into `to_dict`.
        to_dict (``dict``): 
            The dictionary to merge `from_dict` into.

    Returns:
        ``dict``:
            A new dictionary containing the merged data.
    """
    
    # merge first level
    for k in from_dict:
        if k not in to_dict:
            to_dict[k] = from_dict[k]
        else:
            # overwrite merge second level
            for kk in from_dict[k]:
                to_dict[k][kk] = from_dict[k][kk]

    return to_dict
