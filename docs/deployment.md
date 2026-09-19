# Deploying to a Hugging Face Space

The demo runs on a free Hugging Face Space. `deploy/make_space.py` gathers what the Space needs into
`build/space`: the app, the built index, and the Space's own entry point, requirements and settings.

```bash
python deploy/make_space.py
hf auth login                         # a token with write access
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='USER/SPACE', repo_type='space', folder_path='build/space')"
```

`hf upload` doesn't work on a free account: it tries to create the Space first, which the Hub turns
down, so the upload goes through `upload_folder` instead.

The Groq key goes in the Space's settings as a secret named `GROQ_API_KEY`. It must be a secret, not
a variable, because variables are public.

## Why the Space runs on ZeroGPU

Since July 2026 a free account can only run a Gradio Space on ZeroGPU hardware. Docker Spaces and CPU
hardware need a paid plan. ZeroGPU shuts down a Space that hasn't registered a `@spaces.GPU` function
by the time a Gradio app launches, but Anchor never needs a GPU. So `deploy/serve.py`:

- registers one `@spaces.GPU` function that is never called, so no GPU quota is used
- launches an empty Gradio app on a side port, which sends ZeroGPU its start-up report
- keeps the models on the CPU (`ANCHOR_DEVICE=cpu`), since under ZeroGPU torch reports a GPU that
  only exists inside `@spaces.GPU` calls
- serves the FastAPI app on port 7860 as usual

ZeroGPU also needs Python 3.10 or 3.12 and torch 2.8 or later, so the Space pins Python 3.12.12 and
torch 2.8.0 in `deploy/space-readme.md` and `deploy/space-requirements.txt`. The reranker gives the
same scores there as on the machine the evaluation ran on. Both models are downloaded while the Space
builds, so a cold start doesn't wait on them.
