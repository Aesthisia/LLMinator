import os, torch, argparse
from threading import Thread
from typing import Optional

import gradio as gr
from llama_cpp import Llama
from src import quantize
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from core import list_download_models, list_converted_gguf_models, remove_dir, default_repo_id, read_config, update_config
import sys

sys.path.append('./src/llama_cpp/')
sys.path.append('./src/')

cache_dir = os.path.join(os.getcwd(), "models")
saved_models_list = list_download_models(cache_dir)

cache_gguf_dir = os.path.join(os.getcwd(), "src/quantized_model")
saved_gguf_models_list = list_converted_gguf_models(cache_gguf_dir)

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

def init_llm_chain(model_path):
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=6000,
        n_batch=30,
        # temperature=0.9,
        # max_tokens=4095,
        n_parts=1,
        callback_manager=callback_manager, 
        verbose=True)
    
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
                    with gr.Group():
                        repo_id = gr.Textbox(
                            value=default_repo_id,
                            label="Hugging Face Repo",
                            info="Default: openai-community/gpt2"
                        )
                        load_model_btn = gr.Button(
                            value="Load Model",
                            variant="secondary",
                            interactive=True
                        )
                    with gr.Group():
                        execution_provider = gr.Radio(
                            ["cuda", "cpu"], 
                            value=device, 
                            label="Execution providers",
                            info="Select Device"
                        )
                    with gr.Group():
                        saved_models = gr.Dropdown(
                            choices=saved_models_list,
                            max_choices=5, 
                            filterable=True, 
                            label="Saved Models",
                            info="Models available in the disk"
                        )
                        offload_models = gr.ClearButton(
                            value="Remove Cached Models",
                            variant="Secondary",
                            interactive=True,
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
                            ["gguf"],
                            value=["gguf"], 
                            label="Convert Format",
                            interactive=True
                        )
                        download_convert_btn = gr.Button(
                            value="Download Snapshot & Convert",
                            variant="secondary",
                            interactive=True
                        )

                    with gr.Group():
                        converted_models = gr.Dropdown(
                            choices=saved_gguf_models_list,
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

    llm_chain, llm = init_llm_chain(model_path)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def downloadConvertModel(model_repo_id):
        global llm_chain, llm
        if model_repo_id:
            snapshot_download_and_convert_to_gguf(model_repo_id)
            return gr.update(value=""), gr.update(choices=list_converted_gguf_models(cache_gguf_dir))
        else:
            raise gr.Error("Repo can not be empty!")

    def loadModelFromModelsTab(model_repo_id):
        global llm_chain, llm
        model_path = snapshot_download_and_convert_to_gguf(model_repo_id)
        llm_chain, llm = init_llm_chain(model_path)
        update_config(config, repo_id=model_repo_id)
        return gr.update(value=model_repo_id), gr.Tabs(selected="chat")

    def bot(history):
        print("Question: ", history[-1][0])
        output = llm_chain.invoke({"question": history[-1][0]})
        print("stream:", output)
        history[-1][1] = ""
        for character in output:
            print(character)
            history[-1][1] += character
            yield history

    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    download_convert_btn.click(downloadConvertModel, model_repo_id, [model_repo_id, converted_models], queue=False, show_progress="full")
    send_to_chat_btn.click(loadModelFromModelsTab, converted_models, [repo_id, tabs], queue=False, show_progress="full")
    # stop.click(None, None, None, cancels=[submit_event], queue=False)
    # load_model_btn.click(loadModel, repo_id, repo_id, queue=False, show_progress="full")
    # execution_provider.change(fn=updateExecutionProvider, inputs=execution_provider, queue=False, show_progress="full")
    # saved_models.change(loadModel, saved_models, repo_id, queue=False, show_progress="full")
    # offload_models.click(removeModelCache, None, [repo_id, saved_models], queue=False, show_progress="full")

demo.queue()
demo.launch(server_name=args.host, server_port=args.port, share=args.share)