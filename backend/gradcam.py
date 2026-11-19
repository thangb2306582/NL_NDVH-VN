import base64
import cv2
import numpy as np
import tensorflow as tf


def compute_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Tính ma trận heatmap Grad-CAM từ mô hình nhận diện lễ hội.
    img_array: numpy array hoặc tensor shape (1, 224, 224, 3) với giá trị RGB [0, 255]
    """
    try:
        # Tìm backbone model (tìm theo tên kiến trúc, bỏ qua sequential/data_augmentation)
        backbone = None
        for layer in model.layers:
            name_lower = layer.name.lower()
            if any(k in name_lower for k in ["mobilenet", "resnet", "efficient", "densenet", "vgg"]):
                backbone = layer
                break

        if backbone is not None:
            # Tìm lớp tích chập cuối cùng trong backbone (thường là 'out_relu' hoặc 'Conv_1')
            target_layer = None
            for layer in reversed(backbone.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or "relu" in layer.name:
                    target_layer = layer
                    break

            if target_layer is None:
                target_layer = backbone.layers[-1]

            # Model trích xuất activation conv và output của backbone
            bgm = tf.keras.models.Model(
                inputs=backbone.input,
                outputs=[target_layer.output, backbone.output]
            )

            # Tiền xử lý theo backbone
            x_input = np.array(img_array, dtype=np.float32)
            if "mobilenet" in backbone.name.lower():
                x_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(x_input)
            elif "resnet" in backbone.name.lower():
                x_preprocessed = tf.keras.applications.resnet50.preprocess_input(x_input)
            else:
                x_preprocessed = x_input

            with tf.GradientTape() as tape:
                conv_out, bb_out = bgm(x_preprocessed)
                tape.watch(conv_out)

                # Đi qua các lớp sau backbone
                x = bb_out
                gap_layer = model.get_layer("global_average_pooling2d")
                dense_layer = model.get_layer("dense")
                x = gap_layer(x)
                preds = dense_layer(x)

                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_out)
            if grads is None:
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            return heatmap.numpy()

        else:
            # Mô hình phẳng thông thường
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer
                    break
            if last_conv is None:
                return None

            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[last_conv.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_array)
                tape.watch(conv_outputs)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            return heatmap.numpy()

    except Exception as e:
        print(f"[Grad-CAM] Loi tinh toan heatmap: {e}")
        return None


def generate_gradcam_overlay_base64(image_path, model, pred_index=None, alpha=0.45):
    """
    Đọc ảnh từ đường dẫn, sinh heatmap Grad-CAM và trả về Data URI Base64.
    """
    try:
        orig_bgr = cv2.imread(image_path)
        if orig_bgr is None:
            return None

        h, w = orig_bgr.shape[:2]

        # Chuẩn bị ảnh 224x224 RGB
        img_resized = cv2.resize(orig_bgr, (224, 224))
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb.astype(np.float32), 0)

        heatmap = compute_gradcam_heatmap(inp, model, pred_index)
        if heatmap is None:
            return None

        # Resize heatmap về kích thước ảnh gốc
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Bảng màu JET nhiệt độ (Đỏ: quan trọng nhất, Xanh lam: ít quan trọng)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Đè heatmap lên ảnh gốc
        overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)

        # Encode sang JPEG Base64
        _, buffer = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        b64_str = base64.b64encode(buffer).decode("utf-8")

        return f"data:image/jpeg;base64,{b64_str}"

    except Exception as e:
        print(f"[Grad-CAM] Loi tao overlay Base64: {e}")
        return None
