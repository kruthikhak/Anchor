# The Space's entry point, copied to its top level by make_space.py. Docker Spaces are paid now, but
# a free Gradio Space just runs this file and serves whatever answers on port 7860, so it starts the
# FastAPI app there. Gradio gets installed but is never used.
import uvicorn

from app.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
