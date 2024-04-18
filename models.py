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
                    load_model_btn = gr.Button(
                        value="Load Model",
                        variant="secondary",
                        interactive=True)
                with gr.Group():
                    execution_provider = gr.Dropdown(
                    ["a", "b", "c", "d"],
                    value=["a"], 
                    label="Models loader",
                    interactive=True
                )
            with gr.Column():
                with gr.Column():
                    gr.Textbox(
                    value="a",
                    label="Model",
                    info="Default Format: gguf ",
                    interactive=True,
                    scale=1)
                    model_choice = gr.Dropdown(
                    ["a", "b", "c", "d"],
                    value=["a"], 
                    label="Convert",
                    interactive=True
                    )
                    load_model_btn = gr.Button(
                        value="Convert Model",
                        variant="secondary",
                        interactive=True)