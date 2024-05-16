import os, torch, argparse
import gradio as gr
from src import quantize
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from core import list_converted_gguf_models, default_repo_id, read_config, update_config, removeModelFromCache
import sys

sys.path.append('./src/llama_cpp/')
sys.path.append('./src/')

cache_gguf_dir = os.path.join(os.getcwd(), "src/quantized_model")

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

state, config = read_config()
if state == None: 
    config.set('Settings', 'execution_provider', device)
    config.set('Settings', 'repo_id', default_repo_id)
    update_config(config)
else:
    default_repo_id = config.get('Settings', 'repo_id')
    device = config.get('Settings', 'execution_provider')

def snapshot_download_and_convert_to_gguf(repo_id):
    gguf_model_path = quantize.quantize_model(repo_id)
    return gguf_model_path

n_gpu_layers = 0
n_ctx = 6000
n_batch = 30
n_parts = 1
temperature = 0.9
max_tokens = 4095

def init_llm_chain(model_path):    
    if device == "cuda":
        n_gpu_layers = -1
    else:
        n_gpu_layers = 0

    print("inside llm_init_chain function")
    print("n_gpu_layers",n_gpu_layers)
    print("n_ctx", n_ctx)
    print("n_batch", n_batch)
    print("n_parts", n_parts)
    print("temperature", temperature)
    print("max_tokens", max_tokens)

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        temperature=temperature,
        max_tokens=max_tokens,
        n_parts=n_parts,
        callback_manager=callback_manager, 
        verbose=True
    )    
    
    template = """Question: {question}
        Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm
    return llm_chain, llm

def parse_args():
    parser = argparse.ArgumentParser(description='Optional arguments for --host & --port.') 
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host IP to run the server on.')
    parser.add_argument('--port', type=int, default=7860, help='The port to run the server on.')
    parser.add_argument('--share', type=bool, default=False, help='To create a public link.')
    return parser.parse_args()

args = parse_args()

model_path = snapshot_download_and_convert_to_gguf(default_repo_id)

with gr.Blocks(css='style.css') as demo:
    with gr.Tabs(selected="chat") as tabs:
        with gr.Tab("Chat", id="chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    title = gr.Button(
                        value="LLMinator",
                        scale=1,
                        variant="primary",
                        interactive=True,
                        elem_id="title-container"
                    )
                    converted_models_chat = gr.Dropdown(
                        choices=list_converted_gguf_models(cache_gguf_dir),
                        value=default_repo_id,
                        max_choices=5,
                        filterable=True,
                        info="Default: stabilityai/stable-code-instruct-3b",
                        label="Selected Model",
                    )
                    with gr.Group():
                        execution_provider = gr.Radio(
                            ["cuda", "cpu"], 
                            value=device, 
                            label="Execution providers",
                            info="Select Device"
                        )

                with gr.Column(scale=4):
                    with gr.Group():
                        chatbot = gr.Chatbot(elem_id="chatbot-container")
                        msg = gr.Textbox(label="Prompt")
                        stop = gr.Button("Stop")

        with gr.Tab("Models", id="models"):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        model_repo_id = gr.Textbox(
                            value="",
                            label="Hugging Face Repo",
                            info="Default: stabilityai/stable-code-instruct-3b",
                            interactive=True
                        )
                        format_choice = gr.Dropdown(
                            choices=["gguf"],
                            value="gguf",
                            label="Convert Format",
                            interactive=True
                        )
                        download_convert_btn = gr.Button(
                            value="Download Snapshot & Convert",
                            variant="secondary",
                            interactive=True
                        )
                    with gr.Row():
                        with gr.Group():
                            converted_models = gr.Dropdown(
                                choices=list_converted_gguf_models(cache_gguf_dir),
                                value=default_repo_id,
                                max_choices=5,
                                filterable=True,
                                info="gguf models available in the disk",
                                label="Converted Models",
                                interactive=True
                            )
                            send_to_chat_btn = gr.Button(
                                value="Send to Chat",
                                variant="secondary",
                                interactive=True
                            )

                        with gr.Group():
                            saved_gguf_models = gr.Dropdown(
                                choices=list_converted_gguf_models(cache_gguf_dir),
                                max_choices=5,
                                filterable=True,
                                info="gguf models available in the disk",
                                label="Remove Models",
                                interactive=True
                            )
                            remove_model_btn = gr.Button(
                                value="Remove Model",
                                variant="danger",
                                interactive=True
                            )
        with gr.Tab("Configs", id="configs"): 
                    with gr.Row():
                        with gr.Column(elem_id="configs-container"):
                            n_gpu_layers_input = gr.Slider(0, 5000, value=5000, label="n_gpu_layers", visible=torch.cuda.is_available(), interactive= True)
                            n_ctx_input = gr.Slider(100, 6000, value=6000, label="n_ctx", interactive= True)
                            n_batch_input = gr.Slider(1, 512, value=30, label="n_batch", visible=torch.cuda.is_available(), interactive= True)
                            n_parts_input = gr.Slider(0, 10, step=1, value=1, label="n_parts", interactive= True)
                            temperature_input = gr.Slider(0, 1, step=0.1, value=0.9, label="temperature", interactive= True)
                            max_tokens_input = gr.Slider(1, 4095, value=4095, label="max_tokens", interactive= True)

                            with gr.Row():
                                config_update_btn = gr.Button(
                                    value="Update Configs",
                                    variant="primary",
                                    interactive=True
                                )

                                config_reset_btn = gr.Button(
                                    value="Reset Configs",
                                    variant="primary",
                                    interactive=True
                                )

    llm_chain, llm = init_llm_chain(model_path)

    def updateConfigs(n_gpu_layers_input, n_ctx_input, n_batch_input, n_parts_input, temperature_input, max_tokens_input):
        global n_gpu_layers, n_ctx, n_batch, n_parts, temperature, max_tokens, llm_chain, llm
        if torch.cuda.is_available():
            n_gpu_layers = n_gpu_layers_input
            if n_batch_input < n_ctx_input:
                n_batch = n_batch_input
            else:
                raise gr.Error("n_batch should be between 1 and n_ctx")
        else: 
            n_gpu_layers = 0
            n_ctx = 30

        n_ctx = n_ctx_input
        n_parts = n_parts_input
        temperature = temperature_input
        max_tokens = max_tokens_input
        llm_chain, llm = init_llm_chain(model_path)
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.Tabs(selected="chat")

    def resetConfigs():
        global n_gpu_layers, n_ctx, n_batch, n_parts, temperature, max_tokens, llm_chain, llm
        n_gpu_layers = 0
        n_ctx = 6000
        n_batch = 30
        n_parts = 1
        temperature = 0.9
        max_tokens = 4095
        llm_chain, llm = init_llm_chain(model_path)
        return gr.update(value="0"), gr.update(value="6000"), gr.update(value="30"), gr.update(value="1"), gr.update(value="0.9"), gr.update(value="4095")

    def updateExecutionProvider(provider, gguf_model):
        global device
        if provider == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                raise gr.Error("Torch not compiled with CUDA enabled. Please make sure cuda is installed.")

        else:
            device = "cpu"

        update_config(config, execution_provider=provider)
        loadModel(gguf_model)
        return gr.update(value=device)

    def removeModel(model_name):
        removeModelFromCache(model_name)
        return gr.update(choices=list_converted_gguf_models(cache_gguf_dir)), gr.update(choices=list_converted_gguf_models(cache_gguf_dir)), gr.update(choices=list_converted_gguf_models(cache_gguf_dir))

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def downloadConvertModel(model_repo_id):
        if model_repo_id:
            snapshot_download_and_convert_to_gguf(model_repo_id)
            return gr.update(value=""), gr.update(choices=list_converted_gguf_models(cache_gguf_dir)), gr.update(choices=list_converted_gguf_models(cache_gguf_dir)), gr.update(choices=list_converted_gguf_models(cache_gguf_dir))
        else:
            raise gr.Error("Repo can not be empty!")
        
    def loadModel(repo_id):
        global llm_chain, llm
        model_path = snapshot_download_and_convert_to_gguf(repo_id)
        llm_chain, llm = init_llm_chain(model_path)
        update_config(config, repo_id=repo_id)

    def loadModelFromModelsTab(model_repo_id):
        loadModel(model_repo_id)
        return gr.update(value=model_repo_id), gr.Tabs(selected="chat")
    
    def loadModelFromChatTab(repo_id):
        loadModel(repo_id)
        return gr.update(value=repo_id)
    
    def bot(history):
        print("Question: ", history[-1][0])
        output = llm_chain.stream(history[-1][0])
        print("stream:", output)
        history[-1][1] = ""
        for character in output:
            # print(character)
            history[-1][1] += character
            yield history

    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    # stop.click(None, None, None, cancels=[submit_event], queue=False)
    download_convert_btn.click(downloadConvertModel, model_repo_id, [model_repo_id, converted_models_chat, converted_models, saved_gguf_models], queue=False, show_progress="full")
    send_to_chat_btn.click(loadModelFromModelsTab, converted_models, [converted_models_chat, tabs], queue=False, show_progress="full")
    converted_models_chat.change(loadModelFromChatTab, converted_models_chat, converted_models_chat, queue=False, show_progress="full")
    remove_model_btn.click(removeModel, saved_gguf_models, [saved_gguf_models, converted_models_chat, converted_models], queue=False, show_progress="full")
    execution_provider.change(updateExecutionProvider, [execution_provider, converted_models_chat], execution_provider, queue=False, show_progress="full")
    config_update_btn.click(updateConfigs, [n_gpu_layers_input, n_ctx_input, n_batch_input, n_parts_input, temperature_input, max_tokens_input], [n_gpu_layers_input, n_ctx_input, n_batch_input, n_parts_input, temperature_input, max_tokens_input, tabs], queue=False, show_progress="full")
    config_reset_btn.click(resetConfigs, None, [n_gpu_layers_input, n_ctx_input, n_batch_input, n_parts_input, temperature_input, max_tokens_input], queue=False, show_progress="full")

demo.queue()
demo.launch(server_name=args.host, server_port=args.port, share=args.share)