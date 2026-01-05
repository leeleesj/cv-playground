import gradio as gr
import numpy as np
import cv2

# utils
def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb is None:
        return None
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def _resize_image(image_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    if image_bgr is None:
        return None
    h, w = image_bgr.shape[:2]
    if max(h, w) <= long_edge:
        return image_bgr
    if h >= w:
        new_h = long_edge
        new_w = int(w * (long_edge / h))
    else:
        new_w = long_edge
        new_h = int(h * (long_edge / w))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Interface function
def preview_only(image_rgb: np.ndarray, long_edge: int):
    if image_rgb is None:
        return None, "No image provided."

    bgr = _to_bgr(image_rgb)
    bgr = _resize_image(bgr, long_edge)
    rgb = _to_rgb(bgr)

    meta = {
        "status": "ok",
        "shape_hwc": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "long_edge": int(long_edge),
    }
    return rgb, str(meta)

demo = gr.Interface(
    fn=preview_only,
    inputs=[
        gr.Image(type="numpy", label="Input image (RGB)"),
        gr.Slider(
            minimum=64,
            maximum=2048,
            value=1024,
            step=64,
            label="Resize long edge",
        ),
    ],
    outputs=[
        gr.Image(type="numpy", label="Preview (resized)"),
        gr.Textbox(label="Metadata", interactive=False),
    ],
    title="CV Playground (Basics)",
    description="Upload an image and resize it by long edge.",
)

if __name__ == "__main__":
    demo.launch()
