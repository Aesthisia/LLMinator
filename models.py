import gradio as gr
from core import default_repo_id

def create_models_ui():
    with gr.Row():
            with gr.Column():
                with gr.Group():
                    repo_id = gr.Textbox(
                        value=default_repo_id,
                        label="Hugging Face Repo",
                        info="Default: openai-community/gpt2",
                        interactive=True)
                    format_choice = gr.Dropdown(
                    ["gguf"],
                    value=["gguf"], 
                    label="Convert Format",
                    interactive=True
                    )
                    load_model_btn = gr.Button(
                        value="Download Snapshot & convert",
                        variant="secondary",
                        interactive=True)
                with gr.Group():
                    execution_provider = gr.Dropdown(
                    ["a", "b", "c", "d"],
                    value=["a"], 
                    label="Converted Models",
                    interactive=True)

                    load_model_btn = gr.Button(
                        value="Send to chat",
                        variant="secondary",
                        interactive=True)