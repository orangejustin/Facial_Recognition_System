import yaml

def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config