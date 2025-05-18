from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from inference import run_inference  # ✅ 從你寫好的 inference.py 匯入推論函數
import os
import datetime

app = Flask(__name__)
CORS(app)

# ✅ SQLite 資料庫設定
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///records.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

with app.app_context():  # ✅ 確保在 app context 中初始化資料表
    db.create_all()
        
# ✅ 資料表定義
class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_path = db.Column(db.String(200))
    action = db.Column(db.String(20))
    weight = db.Column(db.Integer)
    reps = db.Column(db.Integer)
    datetime = db.Column(db.String(50))

# ✅ 推論 + 上傳影片 API
@app.route("/infer", methods=["POST"])
def infer_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    filename = video.filename
    save_path = os.path.join("videos", filename)
    video.save(save_path)

    result = run_inference(save_path)

    record = Record(
        video_path=result["video_path"],
        action=result["action"],
        weight=result["weight"],
        reps=result["reps"],
        datetime=result["datetime"]
    )
    db.session.add(record)
    db.session.commit()

    return jsonify(result)

# ✅ 查詢資料庫內容 API
@app.route("/records", methods=["GET"])
def get_records():
    records = Record.query.all()
    return jsonify([{
        "id": r.id,
        "video_path": r.video_path,
        "action": r.action,
        "weight": r.weight,
        "reps": r.reps,
        "datetime": r.datetime
    } for r in records])

# ✅ 刪除紀錄 API
@app.route("/records/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    record = Record.query.get(record_id)
    if not record:
        return jsonify({"error": "Record not found"}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Record deleted"})

if __name__ == "__main__":
    app.run(debug=True)
