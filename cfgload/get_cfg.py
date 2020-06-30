import json


def get_cfg(path):
    with open(path) as f:
        cfg = json.load(f)
        params = cfg["parameters"]
        paths = cfg["paths"]
    return params, paths
