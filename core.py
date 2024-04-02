import os, shutil
from configparser import ConfigParser

def format_model_name(directory_name):
    parts = directory_name.split("--")
    return "/".join(parts[1:])

def list_download_models(cache_dir):
    contents = os.listdir(cache_dir)
    directories = [format_model_name(item) for item in contents if os.path.isdir(os.path.join(cache_dir, item)) and item.startswith("models")]
    return directories

def remove_dir(path):
    try:
        shutil.rmtree(os.path.join(path, "/*"))
        print(f"Directory '{path}' successfully removed.")
    except OSError as e:
        print(f"Error: {e.strerror}")

def read_config():
    config = ConfigParser()
    config.read('config.ini')
    return config

def update_config(config):
    with open('config.ini', 'w') as configfile:
        config.write(configfile)