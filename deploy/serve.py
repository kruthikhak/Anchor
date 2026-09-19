# The Space's entry point, copied to its top level by make_space.py.
#
# Free Hugging Face accounts can only run a Gradio Space on ZeroGPU hardware now, and ZeroGPU stops a
# Space that hasn't reported a @spaces.GPU function by the time a Gradio app launches. This app runs on
# the CPU and never needs the GPU, so it registers one function that is never called, launches an empty
# Gradio app on a side port to send that report, keeps the models on the CPU, and serves the FastAPI
# app on port 7860 as before. Run anywhere else, the ZeroGPU part is skipped.
import os

import gradio as gr
import spaces  # before torch, which ZeroGPU requires
import uvicorn


@spaces.GPU
def unused():
    """Never called: it only has to exist when ZeroGPU checks at start-up."""


if __name__ == "__main__":
    os.environ.setdefault("ANCHOR_DEVICE", "cpu")
    if os.environ.get("SPACES_ZERO_GPU"):
        with gr.Blocks() as placeholder:
            pass
        placeholder.launch(server_name="127.0.0.1", server_port=7861, prevent_thread_lock=True, ssr_mode=False)

    from app.server import app

    uvicorn.run(app, host="0.0.0.0", port=7860)
