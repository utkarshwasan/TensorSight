## TensorSight MVP

Real-time image recognition for fruits/vegetables with top-3 confidences, webcam streaming, and analytics (accuracy, confusion matrix, per-class precision/recall/F1, latency, FPS).

### Quickstart

1) Create a virtual environment (recommended).
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Ensure a trained Keras model exists at `Image_classify.keras` (produced by `Image_Class_Model.py`).
4) Launch the Gradio app:
```
python gradio_app.py
```

Gradio will open a local URL. Use the Image tab to upload an image and the Video tab to use your webcam. The Analytics tab evaluates the model on the test set at `Fruits_Vegetables/test`.

### Dataset Structure

Expected directories:
- `Fruits_Vegetables/train/<class_name>/*.jpg|*.png`
- `Fruits_Vegetables/validation/<class_name>/*.jpg|*.png`
- `Fruits_Vegetables/test/<class_name>/*.jpg|*.png`

Class names are inferred from folder names (sorted alphabetically). If training directories are unavailable, a default class list matching `app.py` is used.

### Training

Use `Image_Class_Model.py` to train and save a model. This project does not alter its core fundamentals. It saves `Image_classify.keras` which `gradio_app.py` loads for inference.

### Notes

- The model expects 180x180 RGB inputs. The app resizes frames accordingly and relies on the model's internal `Rescaling(1./255)`.
- Top-3 scores are computed with softmax and displayed in both tabs. Predictions are logged into `logs/predictions.csv`.
- Analytics tab computes test accuracy, per-class precision/recall/F1, and a confusion matrix image.


