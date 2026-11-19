import os
import sys
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# Đảm bảo hiển thị tiếng Việt trên terminal Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =========================== CẤU HÌNH HUẤN LUYỆN ===========================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALID_DIR = BASE_DIR / "dataset" / "val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30


def build_backbone(model_name, input_shape):
    """
    Khởi tạo kiến trúc mạng nền tảng (Backbone) theo tham số lựa chọn:
    - mobilenetv2: Nhẹ, nhanh, phù hợp triển khai thiết bị biên và web
    - resnet50: Cấu trúc Residual kinh điển, độ chính xác cao
    - efficientnetb0: Tối ưu hóa số lượng tham số và hiệu năng
    """
    model_name = model_name.lower()
    
    if model_name == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        # Đóng băng trừ 30 lớp cuối để fine-tuning
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        return base, preprocess_input

    elif model_name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
        base.trainable = True
        for layer in base.layers[:-25]:
            layer.trainable = False
        return base, preprocess_input

    elif model_name == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input
        base = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        return base, preprocess_input

    else:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. Hãy chọn: mobilenetv2, resnet50, hoặc efficientnetb0.")


def train(model_choice="mobilenetv2", epochs=DEFAULT_EPOCHS, train_dataset=None,
          validation_dataset=None, class_names=None):
    print("=" * 70)
    print(f" KHỞI ĐỘNG HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN LỄ HỘI: [{model_choice.upper()}]")
    print("=" * 70)

    if not TRAIN_DIR.exists() or not VALID_DIR.exists():
        print(f"❌ Không tìm thấy thư mục dataset: {TRAIN_DIR} hoặc {VALID_DIR}")
        return

    train_classes = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())
    validation_classes = sorted(path.name for path in VALID_DIR.iterdir() if path.is_dir())
    if len(train_classes) != 14 or train_classes != validation_classes:
        raise ValueError(
            "Dữ liệu phải có đúng 14 thư mục lớp giống nhau trong dataset/train "
            "và dataset/val. "
            f"Train: {len(train_classes)} lớp; Val: {len(validation_classes)} lớp."
        )

    # ================= 1. NẠP DỮ LIỆU (DÙNG CHUNG KHI TRAIN NHIỀU MODEL) =================
    if train_dataset is None or validation_dataset is None:
        print("\n📂 Đang quét tập dữ liệu huấn luyện...")
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            str(TRAIN_DIR), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=True, seed=42
        )

        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            str(VALID_DIR), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode="categorical", shuffle=False
        )

    if class_names is None:
        class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"✅ Phát hiện tổng cộng: {num_classes} lớp lễ hội:")
    for idx, c in enumerate(class_names, 1):
        print(f"   {idx:>2}. {c}")

    # Mỗi mô hình giữ riêng thứ tự nhãn; API dùng manifest của mô hình tốt nhất.
    with open(BASE_DIR / f"labels_{model_choice}.txt", "w", encoding="utf8") as f:
        f.write("\n".join(class_names) + "\n")

    # Tối ưu hóa pipeline dữ liệu
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)

    # ================= 2. DATA AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU) =================
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    # ================= 3. KHỞI TẠO KIẾN TRÚC MÔ HÌNH =================
    base_model, preprocess_func = build_backbone(model_choice, input_shape=(224, 224, 3))

    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    x = data_augmentation(inputs)
    x = preprocess_func(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.35, name="dropout_regularization")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="festival_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"Festival_{model_choice}")

    # ================= 4. BIÊN DỊCH MÔ HÌNH =================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    model.summary()

    # ================= 5. CALLBACKS =================
    checkpoint_name = BASE_DIR / f"best_{model_choice}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_name,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # ================= 6. TIẾN HÀNH HUẤN LUYỆN =================
    print(f"\n🚀 Bắt đầu huấn luyện {epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    # Đánh giá checkpoint tốt nhất trên cùng validation set.
    best_model = tf.keras.models.load_model(checkpoint_name)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in validation_dataset])
    y_prob = best_model.predict(validation_dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    top_k = min(3, num_classes)
    top_k_pred = np.argsort(y_prob, axis=1)[:, -top_k:]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(np.mean([truth in row for truth, row in zip(y_true, top_k_pred)])),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1_score": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
        "confusion_matrix": matrix.tolist(),
    }
    (BASE_DIR / f"evaluation_{model_choice}.json").write_text(
        json.dumps({"model": model_choice, "class_names": class_names, **metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(13, 11))
    image_plot = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image_plot, ax=ax)
    ax.set(title=f"Confusion Matrix - {model_choice}", xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế")
    ax.set_xticks(np.arange(num_classes), class_names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(num_classes), class_names, fontsize=7)
    fig.tight_layout()
    fig.savefig(BASE_DIR / f"confusion_matrix_{model_choice}.png", dpi=180)
    plt.close(fig)
    best_val_accuracy = metrics["accuracy"]
    metadata_path = BASE_DIR / "model_results.json"
    try:
        results = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        results = {}
    results[model_choice] = {
        "model_file": checkpoint_name.name,
        "labels_file": f"labels_{model_choice}.txt",
        "val_accuracy": float(best_val_accuracy),
        "metrics": metrics,
        "epochs_completed": len(history.history.get("accuracy", [])),
    }
    metadata_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Đã lưu {checkpoint_name}; val_accuracy tốt nhất: {best_val_accuracy:.4f}")

    # ================= 7. VẼ VÀ LƯU BIỂU ĐỒ KẾT QUẢ =================
    plt.figure(figsize=(12, 5))

    # Đồ thị Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Huấn luyện (Train)", color="#198754", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Kiểm định (Val)", color="#0d6efd", linewidth=2)
    plt.title(f"Độ chính xác (Accuracy) - {model_choice.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Đồ thị Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Huấn luyện (Train)", color="#dc3545", linewidth=2)
    plt.plot(history.history["val_loss"], label="Kiểm định (Val)", color="#fd7e14", linewidth=2)
    plt.title(f"Hàm mất mát (Loss) - {model_choice.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plot_file = BASE_DIR / f"history_{model_choice}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    print(f"📊 Đã lưu biểu đồ huấn luyện vào '{plot_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện lễ hội văn hóa Việt Nam.")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2",
        choices=["mobilenetv2", "resnet50", "efficientnetb0", "all"],
        help="Lựa chọn kiến trúc mạng nền tảng (Backbone)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Số lượng epochs huấn luyện"
    )

    args = parser.parse_args()
    selected_models = ["mobilenetv2", "resnet50", "efficientnetb0"] if args.model == "all" else [args.model]
    # Nạp dataset một lần, dùng lại tuần tự để giảm thời gian I/O và tránh tải 3 model đồng thời.
    shared_train = tf.keras.preprocessing.image_dataset_from_directory(
        str(TRAIN_DIR), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True, seed=42
    )
    shared_val = tf.keras.preprocessing.image_dataset_from_directory(
        str(VALID_DIR), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False
    )
    with open(BASE_DIR / "labels.txt", "w", encoding="utf8") as labels_file:
        labels_file.write("\n".join(shared_train.class_names) + "\n")
    for selected_model in selected_models:
        tf.keras.backend.clear_session()
        train(selected_model, args.epochs, shared_train, shared_val, shared_train.class_names)

    if args.model == "all":
        results = json.loads((BASE_DIR / "model_results.json").read_text(encoding="utf-8"))
        best_name = max(selected_models, key=lambda name: results[name]["metrics"]["macro_f1"])
        best_model_path = BASE_DIR / results[best_name]["model_file"]
        import shutil
        shutil.copy2(best_model_path, BASE_DIR / "best_model.keras")
        shutil.copy2(BASE_DIR / results[best_name]["labels_file"], BASE_DIR / "labels.txt")
        results["best_model"] = best_name
        (BASE_DIR / "model_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nMô hình tốt nhất theo macro F1: {best_name} ({results[best_name]['metrics']['macro_f1']:.4f})")
