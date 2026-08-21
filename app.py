import os
import sys

import gradio as gr
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import TurnDetector

# Locate default checkpoint or initialize
CHECKPOINT_DIR = "checkpoints"
detector = None


def get_available_checkpoints():
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    return [
        os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")
    ]


def predict_turn(audio, checkpoint_path):
    if audio is None:
        return "Please record or upload an audio clip.", 0.0, 0.0

    sample_rate, audio_data = audio

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Convert to float32 normalized [-1.0, 1.0]
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data /= 32768.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    global detector
    if detector is None or detector.model_path != checkpoint_path:
        if os.path.exists(checkpoint_path):
            detector = TurnDetector(checkpoint_path)
            detector.model_path = checkpoint_path
        else:
            return f"Checkpoint {checkpoint_path} not found. Train a model first!", 0.0, 0.0

    result = detector.predict(audio_data, sample_rate=16000)
    prob = result["probability"]
    latency = result["inference_time_ms"]

    if result["prediction"] == 1:
        status = "✅ Turn Complete (User is done speaking)"
    else:
        status = "⏳ Turn Incomplete (User is pausing / thinking / filler word)"

    return status, prob, f"{latency:.2f} ms"


def create_demo():
    checkpoints = get_available_checkpoints()
    default_ckpt = checkpoints[0] if checkpoints else "checkpoints/exp001_baseline_final.pt"

    with gr.Blocks(title="Shiprocket Turn Detection") as demo:
        gr.Markdown(
            """
            # 🎙️ Real-time Audio Turn Detection Demo
            ### Shiprocket Data Science Challenge
            Detects whether a speaker has **finished their conversational turn** or is **just pausing** (e.g. thinking, using filler words like *'um'*, *'uh'*, *'matlab'*, *'toh'*).
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Speak or Upload Audio (16kHz mono recommended)",
                )
                ckpt_dropdown = gr.Dropdown(
                    choices=checkpoints if checkpoints else [default_ckpt],
                    value=default_ckpt,
                    label="Select Model Checkpoint",
                )
                submit_btn = gr.Button("Analyze Turn Endpoint", variant="primary")

            with gr.Column():
                output_status = gr.Textbox(label="Turn Detection Result")
                output_prob = gr.Slider(
                    minimum=0.0, maximum=1.0, label="Turn Completion Probability", interactive=False
                )
                output_latency = gr.Textbox(label="Inference Latency")

        submit_btn.click(  # type: ignore[reportAttributeAccessIssue]
            fn=predict_turn,
            inputs=[audio_input, ckpt_dropdown],
            outputs=[output_status, output_prob, output_latency],
        )

        gr.Examples(examples=[], inputs=[audio_input])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
