from huggingface_hub import snapshot_download
from src.llama_cpp import convert_hf_to_gguf
import sys

sys.path.append('./llama_cpp/')

def quantize_model(repo_id):
    base_model = "./src/original_model/"
    quantized_path = "./quantized_model/"
    outfile = "./quantized_model/" + repo_id.replace("/", "__") + ".gguf"

    snapshot_download(repo_id=repo_id, local_dir=base_model , local_dir_use_symlinks=False)
    original_model = quantized_path+'/bloom-560m.gguf'

    args = [original_model, "--outtype", "f16", "--outfile", ]
    convert_hf_to_gguf.main(args)

    return outfile


quantize_model("stabilityai/stable-code-instruct-3b")

# !mkdir ./quantized_model/

#python3 ./../../../llama.cpp/convert-hf-to-gguf.py ./original_model/ --outtype f16 --outfile ./quantized_model/bloom-560m.gguf