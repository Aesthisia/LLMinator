import subprocess, os
from huggingface_hub import snapshot_download
from configparser import ConfigParser

config_path = "./configs/config.ini"

def get_py_cmd():
    config = ConfigParser()
    config.read(config_path)
    py_cmd = config.get('Settings', 'py_cmd')
    if "python3" in py_cmd:
        return 'python3'
    else:
        return 'python'

def quantize_model(repo_id):
    original_models_path = "./src/original_model/"
    quantized_path = "./src/quantized_model/"

    repo_id_parts = repo_id.split("/")
    model_folder = f"model--{'--'.join(repo_id_parts)}"
    model_path = original_models_path + model_folder

    outfile = quantized_path + repo_id.replace("/", "__") + ".gguf"
    
    if os.path.isfile(outfile):
        return outfile
    
    snapshot_download(repo_id=repo_id, local_dir=model_path , local_dir_use_symlinks=True)

    command = [
        get_py_cmd(),
        './src/llama_cpp/convert-hf-to-gguf.py',
        model_path,
        '--outtype', 'f16',
        '--outfile', outfile
    ]

    subprocess.run(command, check=True, shell=True)

    return outfile
