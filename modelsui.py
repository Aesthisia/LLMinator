import gradio as gr
from core import default_repo_id

def create_models_ui():
    with gr.Row():
            with gr.Column():
                with gr.Group():
                    repo_id = gr.Textbox(
                        value=default_repo_id,
                        label="Hugging Face Repo",
                        info="Default: stabilityai/stable-code-instruct-3b",
                        interactive=True)
                    format_choice = gr.Dropdown(
                    ["gguf"],
                    value=["gguf"], 
                    label="Convert Format",
                    interactive=True
                    )
                    download_convert_btn = gr.Button(
                        value="Download Snapshot & Convert",
                        variant="secondary",
                        interactive=True)
                with gr.Group():
                    converted_models = gr.Dropdown(
                    ["a", "b", "c", "d"],
                    value=["a"], 
                    label="Converted Models",
                    interactive=True)

                    send_to_chat = gr.Button(
                        value="Send to Chat",
                        variant="secondary",
                        interactive=True)