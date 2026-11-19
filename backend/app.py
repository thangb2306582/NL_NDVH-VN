import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_image
from festival_data import festival_data
from db import db_manager
import json

# ================= Flask Application ================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "model_results.json")
try:
    with open(RESULTS_PATH, "r", encoding="utf-8") as results_file:
        model_results = json.load(results_file)
except (OSError, json.JSONDecodeError):
    model_results = {}


# ===================== Health Check & Status ===========================
@app.route("/", methods=["GET"])
def home():
    db_status = db_manager.get_status()
    return jsonify({
        "status": "online",
        "service": "AI Vietnamese Cultural Festival Recognition API",
        "version": "2.0",
        "database": db_status,
        "total_festivals_supported": len(festival_data)
    })


# ===================== List Festival Reference Data ===================
@app.route("/api/festivals", methods=["GET"])
def get_all_festivals():
    """Trả về danh mục lễ hội và thông tin tóm tắt để người dùng tra cứu."""
    summary_list = []
    for key, item in festival_data.items():
        summary_list.append({
            "key": key,
            "name": item.get("name"),
            "region": item.get("region", "Việt Nam"),
            "place": item.get("place"),
            "time": item.get("time"),
            "meaning": item.get("meaning"),
            "key_visuals": item.get("key_visuals", []),
            "official_links": item.get("official_links", [])
        })
    return jsonify({
        "success": True,
        "total": len(summary_list),
        "recognition_classes": 14,
        "data": summary_list
    })


# ===================== Get Prediction History ===========================
@app.route("/api/history", methods=["GET"])
def get_history():
    """Lấy danh sách các lần nhận diện gần nhất từ MongoDB (hoặc file fallback)."""
    limit = request.args.get("limit", default=20, type=int)
    records = db_manager.get_history(limit=limit)
    return jsonify({
        "success": True,
        "count": len(records),
        "storage": db_manager.get_status()["mode"],
        "data": records
    })


# ===================== Predict Endpoint (POST) ==========================
@app.route("/predict", methods=["POST"])
def predict():
    # Kiểm tra file gửi lên
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "Không tìm thấy tệp ảnh tải lên."
        }), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "message": "Tên tệp rỗng, vui lòng chọn ảnh."
        }), 400

    # Lưu tệp tạm thời
    original_filename = file.filename
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    unique_id = str(uuid.uuid4())
    saved_filename = f"{unique_id}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
    file.save(filepath)

    try:
        # Gọi mô hình nhận diện (kèm tính toán Grad-CAM và Top-3)
        result = predict_image(filepath, return_gradcam=True)

        top1_label = result["label"]
        top1_confidence = result["confidence"]
        is_ood = result["is_ood"]
        heatmap_base64 = result.get("heatmap_base64")

        # Xử lý trường hợp ảnh không hợp lệ / không thuộc dữ liệu huấn luyện (OOD)
        if is_ood:
            record = {
                "id": unique_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_filename": original_filename,
                "top1": {
                    "label": "unrecognized",
                    "name": "Rất tiếc, bức ảnh này chưa hợp lệ. Vui lòng chọn một ảnh khác hoặc chọn một góc chụp rõ nét hơn của lễ hội.",
                    "confidence": top1_confidence
                },
                "top3": [],
                "is_ood": True,
                "is_recognized": False,
                "client_ip": request.remote_addr or "127.0.0.1"
            }
            db_manager.save_prediction(record)

            try:
                os.remove(filepath)
            except Exception:
                pass

            return jsonify({
                "success": True,
                "is_recognized": False,
                "is_ood": True,
                "name": "Rất tiếc, bức ảnh này chưa hợp lệ. Vui lòng chọn một ảnh khác hoặc chọn một góc chụp rõ nét hơn của lễ hội.",
                "label": "unrecognized",
                "confidence": top1_confidence,
                "message": "Rất tiếc, bức ảnh này chưa hợp lệ. Vui lòng chọn một ảnh khác hoặc chọn một góc chụp rõ nét hơn của lễ hội.",
                "ood_warning": "Rất tiếc, bức ảnh này chưa hợp lệ. Vui lòng chọn một ảnh khác hoặc chọn một góc chụp rõ nét hơn của lễ hội.",
                "top3": [],
                "summary": None,
                "heatmap": None,
                "models_comparison": None
            })

        # Bổ sung tên tiếng Việt cho Top-3
        enriched_top3 = []
        for item in result["top3"]:
            lbl = item["label"]
            fest_info = festival_data.get(lbl, {})
            enriched_top3.append({
                "label": lbl,
                "name": fest_info.get("name", lbl),
                "confidence": item["confidence"],
                "region": fest_info.get("region", "")
            })

        # Tra cứu thông tin chi tiết lễ hội quán quân
        info = festival_data.get(top1_label, {
            "name": top1_label,
            "region": "Chưa xác định",
            "description": "Đang cập nhật thông tin chi tiết.",
            "time": "Chưa rõ",
            "place": "Chưa rõ",
            "people": "Chưa rõ",
            "meaning": "Chưa rõ",
            "key_visuals": [],
            "official_links": []
        })

        # Tạo bản ghi lưu trữ lịch sử
        record = {
            "id": unique_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_filename": original_filename,
            "top1": {
                "label": top1_label,
                "name": info["name"],
                "confidence": top1_confidence
            },
            "top3": enriched_top3,
            "is_ood": is_ood,
            "client_ip": request.remote_addr or "127.0.0.1"
        }

        # Lưu vào Database (MongoDB hoặc Fallback JSON)
        db_manager.save_prediction(record)

        # Xóa file ảnh tạm sau khi xử lý
        try:
            os.remove(filepath)
        except Exception:
            pass

        # Trả về kết quả hoàn chỉnh
        return jsonify({
            "success": True,
            "is_ood": is_ood,
            "ood_warning": "Độ tin cậy dưới ngưỡng quy định. Bức ảnh có thể không thuộc các lễ hội trong hệ thống hoặc chất lượng ảnh không rõ nét." if is_ood else None,
            "label": top1_label,
            "name": info["name"],
            "region": info.get("region", "Việt Nam"),
            "confidence": top1_confidence,
            "top3": enriched_top3,
            "all_scores": result["all_scores"],
            "summary": {
                "description": info.get("description"),
                "time": info.get("time"),
                "place": info.get("place"),
                "people": info.get("people"),
                "meaning": info.get("meaning"),
                "key_visuals": info.get("key_visuals", []),
                "official_links": info.get("official_links", [])
            },
            "heatmap": heatmap_base64,
            "models_comparison": result.get("models_comparison")
        })

    except Exception as e:
        # Dọn dẹp ảnh nếu xảy ra lỗi
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

        print(f"[API Error] /predict gap loi: {e}")
        return jsonify({
            "success": False,
            "message": f"Lỗi trong quá trình nhận diện: {str(e)}"
        }), 500


# ================================ Entry Point ====================
if __name__ == "__main__":
    print("[Server] Khoi dong AI Festival Recognition API tai port 8000...")
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
