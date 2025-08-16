import os
import time
import csv
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import tempfile

import numpy as np
import tensorflow as tf

import gradio as gr
import cv2
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Constants and global state
# -----------------------------------------------------------------------------
IMG_HEIGHT = 180
IMG_WIDTH = 180
MODEL_PATH = "Image_classify.keras"
TRAIN_DIR = os.path.join("Fruits_Vegetables", "train")
TEST_DIR = os.path.join("Fruits_Vegetables", "test")

# Lightweight custom styling
CUSTOM_CSS = """
.header {
  background: linear-gradient(135deg, #0ea5e9, #22c55e);
  padding: 16px 18px;
  border-radius: 14px;
  color: white;
}
.pill {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.15);
  margin-right: 6px;
  font-size: 12px;
}
.card {
  background: var(--neutral-800);
  border: 1px solid var(--neutral-600);
  border-radius: 12px;
  padding: 12px;
}
.score-row { display:flex; align-items:center; gap:10px; margin: 8px 0; }
.score-label { min-width: 140px; font-weight: 600; }
.score-bar { flex:1; height: 10px; background: var(--neutral-700); border-radius: 8px; overflow: hidden; }
.score-fill { height: 100%; background: linear-gradient(90deg,#34d399,#10b981); }
.big-metric { font-size: 36px; font-weight: 800; }
"""


def ensure_logs_dir() -> str:
    logs_dir = os.path.join("logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_class_names_from_dir(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        # Fallback to class list in app.py if directory is not present
        return ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
                'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
                'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
                'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
                'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
    class_names = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    class_names.sort()
    return class_names


CLASS_NAMES: List[str] = get_class_names_from_dir(TRAIN_DIR)


def load_keras_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train and save as '{MODEL_PATH}'.")
    model = tf.keras.models.load_model(model_path)
    return model


MODEL = load_keras_model(MODEL_PATH)


def preprocess_image(np_image: np.ndarray) -> tf.Tensor:
    # Expect HxWxC (RGB). Convert to PIL then resize to match training.
    if np_image.dtype != np.uint8:
        # Normalize float images to [0,255] and cast
        arr = np.clip(np_image, 0, 255)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        np_image = arr.astype(np.uint8)
    image = Image.fromarray(np_image)
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_np = np.array(image)
    # Model includes a Rescaling(1./255) layer, so keep 0-255 values here
    image_np = image_np.astype(np.uint8)
    batched = np.expand_dims(image_np, axis=0)
    return tf.convert_to_tensor(batched)


def predict_topk(np_image: np.ndarray, k: int = 3) -> Tuple[List[str], List[float], Dict[str, float]]:
    input_tensor = preprocess_image(np_image)
    logits = MODEL(input_tensor, training=False)
    probs = tf.nn.softmax(logits, axis=-1)[0]
    values, indices = tf.math.top_k(probs, k=min(k, probs.shape[-1]))

    top_labels = [CLASS_NAMES[int(i)] for i in indices.numpy().tolist()]
    top_scores = values.numpy().astype(float).tolist()
    label_scores = {lbl: float(score) for lbl, score in zip(top_labels, top_scores)}
    return top_labels, top_scores, label_scores


def overlay_predictions(frame_rgb: np.ndarray, labels: List[str], scores: List[float]) -> np.ndarray:
    # Convert to BGR for OpenCV drawing, then back to RGB for display
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    y0 = 25
    dy = 25
    for rank, (lbl, sc) in enumerate(zip(labels, scores), start=1):
        text = f"{rank}. {lbl}: {sc*100:.1f}%"
        cv2.putText(frame_bgr, text, (10, y0 + (rank - 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def log_prediction(source: str, top_labels: List[str], top_scores: List[float]) -> None:
    logs_dir = ensure_logs_dir()
    csv_path = os.path.join(logs_dir, "predictions.csv")
    header = [
        "timestamp",
        "source",
        "top1_label",
        "top1_score",
        "top2_label",
        "top2_score",
        "top3_label",
        "top3_score",
    ]
    # Use timezone-aware UTC timestamps
    row = [
        datetime.now(timezone.utc).isoformat(),
        source,
        top_labels[0] if len(top_labels) > 0 else "",
        f"{top_scores[0]:.6f}" if len(top_scores) > 0 else "",
        top_labels[1] if len(top_labels) > 1 else "",
        f"{top_scores[1]:.6f}" if len(top_scores) > 1 else "",
        top_labels[2] if len(top_labels) > 2 else "",
        f"{top_scores[2]:.6f}" if len(top_scores) > 2 else "",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Gradio functions and helpers
# -----------------------------------------------------------------------------
def render_top3_html(labels: List[str], scores: List[float]) -> str:
    if not labels or not scores:
        return "<div class='card'>No predictions.</div>"
    rows = []
    for lbl, sc in zip(labels, scores):
        pct = max(0.0, min(100.0, float(sc) * 100.0))
        rows.append(
            f"<div class='score-row'>"
            f"<div class='score-label'>{lbl}</div>"
            f"<div class='score-bar'><div class='score-fill' style='width:{pct:.1f}%'></div></div>"
            f"<div style='min-width:56px;text-align:right'>{pct:.1f}%</div>"
            f"</div>"
        )
    return "<div class='card'>" + "".join(rows) + "</div>"
def classify_image(image: np.ndarray):
    try:
        if image is None:
            return "<div class='card'>No image provided.</div>", {"error": 1.0}, None, "No image provided."

        top_labels, top_scores, label_scores = predict_topk(image, k=3)

        # Log
        log_prediction("image_upload", top_labels, top_scores)

        # Annotate for preview
        annotated = overlay_predictions(image, top_labels, top_scores)

        # Convert label_scores to fixed 3 entries for consistent UI ordering
        top3_dict = {lbl: float(score) for lbl, score in label_scores.items()}
        return (
            render_top3_html(top_labels, top_scores),
            top3_dict,
            annotated,
            f"Top-1: {top_labels[0]} ({top_scores[0]*100:.1f}%)",
        )
    except Exception as e:
        return (
            f"<div class='card'>Error: {str(e)}</div>",
            {"error": 1.0},
            None,
            "Error",
        )


def stream_webcam(frame: np.ndarray):
    try:
        if frame is None:
            yield None, {"error": 1.0}, "<div class='card'>No frame.</div>", "No frame."
            return

        top_labels, top_scores, label_scores = predict_topk(frame, k=3)

        annotated = overlay_predictions(frame, top_labels, top_scores)

        # Log intermittently is removed alongside latency tracking

        yield (
            annotated,
            {lbl: float(score) for lbl, score in zip(top_labels, top_scores)},
            render_top3_html(top_labels, top_scores),
            f"Top-1: {top_labels[0]} ({top_scores[0]*100:.1f}%)",
        )
    except Exception as e:
        yield None, {"error": 1.0}, f"<div class='card'>Error: {str(e)}</div>", "Error"


def run_evaluation():
    try:
        if not os.path.isdir(TEST_DIR):
            return (
                0.0,
                None,
                pd.DataFrame({"error": ["Test directory not found: " + TEST_DIR]}),
            )

        ds = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            shuffle=False,
            batch_size=32,
        )
        class_names_eval = ds.class_names

        y_true = []
        y_pred = []

        for batch_images, batch_labels in ds:
            logits = MODEL(batch_images, training=False)
            probs = tf.nn.softmax(logits, axis=-1)
            preds = tf.argmax(probs, axis=-1)
            y_true.extend(batch_labels.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())

        acc = float(accuracy_score(y_true, y_pred))

        # Per-class metrics
        labels_idx = list(range(len(class_names_eval)))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_idx, zero_division=0
        )
        report_df = pd.DataFrame(
            {
                "class": class_names_eval,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

        # Confusion matrix figure
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names_eval,
            yticklabels=class_names_eval,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix (Test Set)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        # Convert figure to numpy image for Gradio (robust for recent Matplotlib)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
        # Drop alpha channel
        cm_img = rgba[:, :, :3].copy()
        plt.close(fig)

        return acc, cm_img, report_df
    except Exception as e:
        return 0.0, None, pd.DataFrame({"error": [str(e)]})


def build_interface() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="gray")
    with gr.Blocks(title="TensorSight - Real-Time Image Recognition", theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML("""
        <div class='header'>
          <div style='font-size:22px;font-weight:800;'>🧠 TensorSight</div>
          <div style='opacity:0.95;margin-top:6px;'>Real-time fruits & vegetables recognition with top‑3 confidences and analytics.</div>
          <div style='margin-top:8px;'>
            <span class='pill'>Webcam & Video</span>
            <span class='pill'>Analytics</span>
          </div>
        </div>
        """)

        with gr.Tab("🖼️ Image"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Upload Image", height=320)
                with gr.Column(scale=1, min_width=140):
                    img_btn = gr.Button("🚀 Classify", variant="primary")
            gr.Examples(
                examples=[["Apple.jpg"], ["Banana.jpg"], ["corn.jpg"], ["cabbage.jpg"]],
                inputs=[img_in],
            )
            with gr.Row():
                top3_html = gr.HTML(label="Top‑3 Confidence")
                label_out = gr.Label(num_top_classes=3, label="Predictions (Table)")
            with gr.Row():
                preview_out = gr.Image(label="Annotated Preview", height=320)
            top1_text = gr.Textbox(label="Top‑1", interactive=False)
            img_btn.click(
                fn=classify_image,
                inputs=[img_in],
                outputs=[top3_html, label_out, preview_out, top1_text],
            )

        with gr.Tab("🎥 Video"):
            with gr.Tabs():
                with gr.TabItem("Webcam"):
                    webcam_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam", height=320)
                    with gr.Row():
                        frame_out = gr.Image(label="Annotated Stream", height=320)
                        label_stream_out = gr.Label(num_top_classes=3, label="Predictions (current frame)")
                    top3_stream_html = gr.HTML(label="Top‑3 Confidence (current frame)")
                    top1_stream = gr.Textbox(label="Top‑1", interactive=False)

                    webcam_in.stream(
                        fn=stream_webcam,
                        inputs=[webcam_in],
                        outputs=[frame_out, label_stream_out, top3_stream_html, top1_stream],
                    )

                with gr.TabItem("Video File"):
                    gr.Markdown("Upload a video file to generate an annotated output video with predictions and metrics.")
                    video_in = gr.Video(label="Upload Video File")
                    process_btn = gr.Button("🎬 Process Video", variant="primary")
                    video_out = gr.Video(label="Annotated Video Output")
                    vid_frames = gr.Number(label="Total Frames", interactive=False)

                    def process_video(video_path: str):
                        if not video_path or not os.path.exists(video_path):
                            return None, 0.0, 0.0, 0

                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            return None, 0.0, 0.0, 0

                        # Prepare writer
                        fps_input = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        tmp_dir = ensure_logs_dir()
                        out_path = os.path.join(tmp_dir, f"annotated_{int(time.time())}.mp4")
                        writer = cv2.VideoWriter(out_path, fourcc, fps_input, (width, height))

                        frames = 0
                        try:
                            while True:
                                ret, frame_bgr = cap.read()
                                if not ret:
                                    break
                                frames += 1
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                labels, scores, _ = predict_topk(frame_rgb, k=3)
                                annotated_rgb = overlay_predictions(frame_rgb, labels, scores)
                                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                                writer.write(annotated_bgr)
                        finally:
                            cap.release()
                            writer.release()

                        return out_path, frames

                    process_btn.click(
                        fn=process_video,
                        inputs=[video_in],
                        outputs=[video_out, vid_frames],
                    )

        with gr.Tab("📈 Analytics"):
            gr.Markdown("Click Run to evaluate the model on the test set and view metrics.")
            run_eval_btn = gr.Button("📊 Run Evaluation", variant="primary")
            acc_out = gr.Number(label="Test Accuracy")
            cm_out = gr.Image(label="Confusion Matrix")
            report_out = gr.Dataframe(label="Per-class Precision/Recall/F1", interactive=False)
            run_eval_btn.click(fn=run_evaluation, inputs=[], outputs=[acc_out, cm_out, report_out])

            # Logs download
            def get_logs_file():
                p = os.path.join(ensure_logs_dir(), "predictions.csv")
                return p if os.path.exists(p) else None

            with gr.Row():
                logs_btn = gr.Button("⬇️ Refresh Logs")
                logs_file = gr.File(label="Prediction Logs (CSV)")
            logs_btn.click(fn=get_logs_file, inputs=[], outputs=[logs_file])

        gr.Markdown("""
        <div style='opacity:0.7;margin-top:8px'>Tip: Use the Image tab for quick checks; switch to Video for live streams. Analytics helps benchmark on your test set.</div>
        """)

    return demo


if __name__ == "__main__":
    ui = build_interface()
    ui.queue().launch()


