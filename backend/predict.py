import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import keras

# =================== Keras Deserialization Compatibility Patch ===================
_orig_glorot_init = keras.initializers.GlorotUniform.__init__
keras.initializers.GlorotUniform.__init__ = lambda self, *args, **kwargs: _orig_glorot_init(
    self, *args, **{k: v for k, v in kwargs.items() if k not in ("input_axes", "output_axes")}
)

# =============================== Load Model & Labels ============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# Load labels
with open(LABELS_PATH, "r", encoding="utf8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

# Refuse to associate a model's output indices with unrelated labels. The
# checked-in model currently has five outputs while the dataset has fourteen.
try:
    with open(os.path.join(BASE_DIR, "model_results.json"), encoding="utf-8") as f:
        model_results = json.load(f)
except (OSError, json.JSONDecodeError):
    model_results = {}

MODEL_SPECS = {
    "mobilenetv2": ("MobileNetV2", "best_mobilenetv2.keras"),
    "resnet50": ("ResNet50", "best_resnet50.keras"),
    "efficientnetb0": ("EfficientNetB0", "best_efficientnetb0.keras"),
}
trained_models = {}
for model_id, (model_name, default_file) in MODEL_SPECS.items():
    entry = model_results.get(model_id, {})
    path = os.path.join(BASE_DIR, entry.get("model_file", default_file))
    labels_path = os.path.join(BASE_DIR, entry.get("labels_file", f"labels_{model_id}.txt"))
    if not os.path.isfile(path) or not os.path.isfile(labels_path):
        continue
    try:
        model_labels = [line.strip() for line in open(labels_path, encoding="utf-8") if line.strip()]
        candidate = tf.keras.models.load_model(path)
        if model_labels == labels and int(candidate.output_shape[-1]) == len(labels):
            trained_models[model_id] = candidate
    except Exception as exc:
        print(f"[Predict] Không nạp được {model_name}: {exc}")

best_model_id = model_results.get("best_model")
if best_model_id not in trained_models and trained_models:
    best_model_id = max(trained_models, key=lambda key: model_results.get(key, {}).get("metrics", {}).get("macro_f1", 0))
model = trained_models.get(best_model_id)
if model is None:
    raise RuntimeError("Chưa có mô hình 14 lớp đã huấn luyện và báo cáo nhãn hợp lệ. Hãy train trước khi chạy API.")

# Import Grad-CAM
from gradcam import generate_gradcam_overlay_base64

CONFIDENCE_THRESHOLD = 0.55  # Ngưỡng phát hiện ảnh không phải lễ hội (OOD)


def predict_image(image_path, return_gradcam=True):
    """D? ?o?n b?ng m? h?nh ???c ch?n theo macro F1 tr?n t?p validation."""
    img = image.load_img(image_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    probabilities = model.predict(img_batch, verbose=0)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    top1_index = int(sorted_indices[0])
    confidence = float(probabilities[top1_index])
    top1_label = labels[top1_index]
    top3_results = [
        {"label": labels[int(i)], "confidence": round(float(probabilities[i]) * 100, 2),
         "probability": float(probabilities[i])}
        for i in sorted_indices[:min(3, len(labels))]
    ]

    models_list = []
    for model_id in trained_models:
        metrics = model_results[model_id]["metrics"]
        models_list.append({
            "id": model_id,
            "name": MODEL_SPECS[model_id][0],
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "top3_accuracy": round(metrics["top3_accuracy"] * 100, 2),
            "macro_precision": round(metrics["macro_precision"] * 100, 2),
            "macro_recall": round(metrics["macro_recall"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"] * 100, 2),
            "weighted_f1": round(metrics["weighted_f1"] * 100, 2),
            "is_best": model_id == best_model_id,
        })

    is_ood = confidence < CONFIDENCE_THRESHOLD
    heatmap_base64 = None
    if return_gradcam and not is_ood:
        try:
            heatmap_base64 = generate_gradcam_overlay_base64(
                image_path=image_path, model=model, pred_index=top1_index, alpha=0.45
            )
        except Exception as exc:
            print(f"[Predict] Kh?ng th? t?o Grad-CAM: {exc}")

    return {
        "label": top1_label,
        "confidence": round(confidence * 100, 2),
        "top3": top3_results,
        "all_scores": {labels[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(labels))},
        "is_ood": is_ood,
        "confidence_threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
        "heatmap_base64": heatmap_base64,
        "models_comparison": {
            "best_model_name": MODEL_SPECS[best_model_id][0],
            "best_model_id": best_model_id,
            "selection_metric": "macro_f1",
            "models": models_list,
        },
    }


if __name__ == "__main__":
    test_img = os.path.join("dataset", "train", "dua_bo_bay_nui", "DB_1.jpg")
    if os.path.exists(test_img):
        res = predict_image(test_img)
        print("Best Model:", res["models_comparison"]["best_model_name"], res["models_comparison"]["best_confidence"], "%")
        print("Models list:", res["models_comparison"]["models"])
