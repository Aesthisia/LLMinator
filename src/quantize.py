import subprocess
from huggingface_hub import snapshot_download
# from src.llama_cpp import convert_hf_to_gguf
# import sys

#sys.path.append('./llama_cpp/')

def quantize_model(repo_id):
    base_model = "./src/original_model/"
    quantized_path = "./src/quantized_model/"
    outfile = quantized_path + repo_id.replace("/", "__") + ".gguf"

    snapshot_download(repo_id=repo_id, local_dir=base_model , local_dir_use_symlinks=True)
    original_model = quantized_path+'/bloom-560m.gguf'

    command = [
        'python3', 
        './src/llama_cpp/convert-hf-to-gguf.py',
        base_model,
        '--outtype', 'f16',
        '--outfile', outfile
    ]

    # Run the command
    subprocess.run(command, check=True)

    return outfile


#quantize_model("stabilityai/stable-code-instruct-3b")

# !mkdir ./quantized_model/

#