import os
import json
import sys
from datetime import datetime

# Đảm bảo UTF-8 trên Windows console nếu có thể
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cố gắng import pymongo nếu đã cài đặt
try:
    from mongo_config import (
        MONGO_URI,
        DB_NAME,
        COLLECTION_NAME,
        connect_mongodb,
    )
    PYMONGO_AVAILABLE = True
except ImportError:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = "festival_db"
    COLLECTION_NAME = "prediction_history"
    connect_mongodb = None
    PYMONGO_AVAILABLE = False

FALLBACK_FILE = os.path.join(os.path.dirname(__file__), "history_fallback.json")


class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.use_mongo = False
        self._init_connection()

    def _init_connection(self):
        """Khởi tạo kết nối tới MongoDB, nếu không được thì kích hoạt chế độ Fallback."""
        if not PYMONGO_AVAILABLE:
            print("[Database] Thu vien pymongo chua san sang. Dung che do JSON Fallback.")
            self.use_mongo = False
            return

        try:
            # Timeout 1500ms để không làm chậm server nếu chưa bật MongoDB
            self.client, self.db, self.collection = connect_mongodb()
            self.use_mongo = True
            print(f"[Database] Ket noi thanh cong MongoDB tai: {MONGO_URI} (DB: {DB_NAME})")
        except Exception:
            self.use_mongo = False
            print("[Database] Chua phat hien MongoDB localhost:27017 dang chay.")
            print(f"[Database] Dang su dung che do luu tru du phong: {os.path.basename(FALLBACK_FILE)}")

    def save_prediction(self, record):
        """Lưu một bản ghi nhận diện."""
        # Thử kết nối lại MongoDB nếu lần trước chưa kết nối được
        if not self.use_mongo and PYMONGO_AVAILABLE:
            try:
                self._init_connection()
            except Exception:
                pass

        if self.use_mongo and self.collection is not None:
            try:
                self.collection.insert_one(record)
                return True
            except Exception as e:
                print(f"[Database] Loi ghi MongoDB: {e}. Ghi tam vao JSON.")

        # Chế độ Fallback: Lưu vào file JSON
        try:
            records = []
            if os.path.exists(FALLBACK_FILE):
                with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                    try:
                        records = json.load(f)
                    except Exception:
                        records = []

            # Loại bỏ _id nếu có từ Mongo để lưu JSON hợp lệ
            clean_record = {k: v for k, v in record.items() if k != "_id"}
            records.insert(0, clean_record)

            # Giới hạn lưu 200 bản ghi gần nhất
            records = records[:200]

            with open(FALLBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[Database] Loi ghi JSON fallback: {e}")
            return False

    def get_history(self, limit=20):
        """Lấy danh sách lịch sử nhận diện gần nhất."""
        if self.use_mongo and self.collection is not None:
            try:
                cursor = self.collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                print(f"[Database] Loi doc MongoDB: {e}. Chuyen sang doc JSON.")

        # Đọc từ file JSON fallback
        if os.path.exists(FALLBACK_FILE):
            try:
                with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data[:limit]
            except Exception as e:
                print(f"[Database] Loi doc JSON fallback: {e}")
                return []
        return []

    def get_status(self):
        """Trả về trạng thái hiện tại của cơ sở dữ liệu."""
        return {
            "mode": "mongodb" if self.use_mongo else "json_fallback",
            "mongodb_connected": self.use_mongo,
            "target_uri": MONGO_URI,
            "fallback_file": os.path.basename(FALLBACK_FILE)
        }


# Khởi tạo một thể hiện dùng chung cho ứng dụng
db_manager = DatabaseManager()
