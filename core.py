import os, shutil
from configparser import ConfigParser

default_repo_id = "stabilityai/stable-code-instruct-3b"
default_repo_id_parts = default_repo_id.split("/")
default_model_folder = f"models--{'--'.join(default_repo_id_parts)}"

def format_model_name(directory_name):
    parts = directory_name.split("--")
    return "/".join(parts[1:])

def list_download_models(cache_dir):
    contents = os.listdir(cache_dir)
    directories = [format_model_name(item) for item in contents if os.path.isdir(os.path.join(cache_dir, item)) and item.startswith("models")]
    return directories

def remove_dir(path):
    try:
        for model in os.listdir(path):
            if model != default_model_folder:
                model_path = os.path.join(path, model)
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
        print("successfully removed cached models!")
    except OSError as e:
        print(f"Error: {e.strerror}")

def read_config():
    config = ConfigParser()
    config.read('config.ini')
    return config

def update_config(config):
    with open('config.ini', 'w') as configfile:
        config.write(configfile)