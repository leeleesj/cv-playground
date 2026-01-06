import gradio as gr
import numpy as np

from src.operators import laplacian, sobel

# ----------------------------
# Wrappers for UI callbacks
# ----------------------------
def run_laplacian(image_rgb: np.ndarray, ksize: int, use_abs: bool):
    out, meta = laplacian(image_rgb, ksize=int(ksize), use_abs=bool(use_abs))
    return out, meta


def run_sobel(image_rgb: np.ndarray, ksize: int, mode: str):
    out, meta = sobel(image_rgb, ksize=int(ksize), mode=mode)
    return out, meta


def run_fft_placeholder(image_rgb: np.ndarray):
    # TODO
    if image_rgb is None:
        return None, {"status": "no_image"}
    h, w = image_rgb.shape[:2]
    return image_rgb, {"status": "ok", "op": "fft", "note": "placeholder", "shape": [h, w]}


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="CV Playground") as demo:
    gr.Markdown(
        """
# CV Playground
This is a simple tool to visualize and inspect metadata of various computer vision operators.
        """.strip()
    )

    with gr.Row():
        # LEFT: shared input
        with gr.Column(scale=1, min_width=360):
            inp = gr.Image(type="numpy", label="Input image (RGB)")
            gr.Examples(
                examples=[
                    "assets/example/lisboa_tail.jpg",
                    "assets/example/lisboa.jpg",
                ],
                inputs=inp,
                label="Example images",
            )

        # RIGHT: results (tabs)
        with gr.Column(scale=1, min_width=520):
            with gr.Tabs():
                # ---------------- Laplacian ----------------
                with gr.TabItem("Laplacian"):
                    with gr.Row():
                        lap_ksize = gr.Radio(
                            choices=[1, 3, 5],
                            value=3,
                            label="Kernel size (ksize)",
                        )
                        lap_abs = gr.Checkbox(value=True, label="abs() for visualization")

                    lap_out = gr.Image(type="numpy", label="Laplacian output", interactive=False)
                    lap_meta = gr.JSON(label="Metadata")

                    lap_btn = gr.Button("Run Laplacian", variant="primary")

                # ---------------- Sobel ----------------
                with gr.TabItem("Sobel"):
                    with gr.Row():
                        sobel_ksize = gr.Radio(
                            choices=[1, 3, 5],
                            value=3,
                            label="Kernel size (ksize)",
                        )
                        sobel_mode = gr.Radio(
                            choices=["magnitude", "x", "y"],
                            value="magnitude",
                            label="Mode",
                        )

                    sobel_out = gr.Image(type="numpy", label="Sobel output", interactive=False)
                    sobel_meta = gr.JSON(label="Metadata")

                    sobel_btn = gr.Button("Run Sobel", variant="primary")

                # ---------------- FFT (placeholder) ----------------
                with gr.TabItem("FFT"):
                    gr.Markdown("TODO: FFT magnitude (log), radial energy, high-frequency ratio")
                    fft_out = gr.Image(type="numpy", label="FFT output (placeholder)", interactive=False)
                    fft_meta = gr.JSON(label="Metadata")

                    fft_btn = gr.Button("Run FFT (placeholder)", variant="primary")

    # --- Button triggers (explicit runs) ---
    lap_btn.click(
        fn=run_laplacian,
        inputs=[inp, lap_ksize, lap_abs],
        outputs=[lap_out, lap_meta],
    )

    sobel_btn.click(
        fn=run_sobel,
        inputs=[inp, sobel_ksize, sobel_mode],
        outputs=[sobel_out, sobel_meta],
    )

    fft_btn.click(
        fn=run_fft_placeholder,
        inputs=[inp],
        outputs=[fft_out, fft_meta],
    )


if __name__ == "__main__":
    demo.launch()
