import os

def format_model_name(directory_name):
    parts = directory_name.split("--")
    return "/".join(parts[1:])

def list_download_models(cache_dir):
    contents = os.listdir(cache_dir)
    directories = [item for item in contents if os.path.isdir(os.path.join(cache_dir, item)) and item.startswith("models")]
    return directories


