import os
import cv2
import yaml
import numpy as np
from collections import deque
from statistics import mode
from ultralytics import YOLO
from datetime import datetime

# === 常數與模型 ===
POSE_MODEL_PATH = "models/yolov8n-pose.pt"
PLATE_YAML = "config/plate.yaml"
PLATE_MODEL = "models/plate.pt"
BARBELL_MODEL = "models/barbell.pt"
OUTPUT_DIR = "videos/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEIGHT_MAPPING = {
    0: 25, 1: 20, 2: 15, 3: 10, 4: 5,
    5: 2.5, 6: 2.0, 7: 1.5, 8: 1.0, 9: 0.5,
    10: 2.5, 11: 20  # barbell weight = 20kg
}

# === 公用工具 ===
def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)["names"]

pose_model = YOLO(POSE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL)
barbell_model = YOLO(BARBELL_MODEL)
class_names = load_class_names(PLATE_YAML)

# === PlateProcessor 類別 ===
class PlateProcessor:
    def __init__(self, weight_map, barbell_weight=20):
        self.cx = None
        self.weight_map = weight_map
        self.barbell_weight = barbell_weight
        self.left_ids = []
        self.right_ids = []
        self.confirmed_state = "Unverified"
        self.message = "--"
        self.total_weight = 0
        self.state_buffer = deque(maxlen=5)
        self.weight_buffer = deque(maxlen=5)

    def _is_descending(self, id_list):
        weights = [self.weight_map.get(cls, 0) for cls in id_list]
        return all(w1 >= w2 for w1, w2 in zip(weights, weights[1:]))

    def update(self, plates, barbell_box):
        self.cx = (barbell_box[0] + barbell_box[2]) // 2
        left = [(x, cls) for (x, cls) in plates if x < self.cx]
        right = [(x, cls) for (x, cls) in plates if x >= self.cx]
        left = sorted(left, key=lambda p: abs(p[0] - self.cx))
        right = sorted(right, key=lambda p: abs(p[0] - self.cx))
        self.left_ids = [cls for _, cls in left]
        self.right_ids = [cls for _, cls in right]

        if len(self.left_ids) != len(self.right_ids):
            current_state = "Mismatch"
            current_message = "Plates not symmetric"
        elif self.left_ids != self.right_ids:
            current_state = "Mismatch"
            current_message = f"Left not equal to Right: {self.left_ids} not equal to {self.right_ids}"
        elif not self._is_descending(self.left_ids) or not self._is_descending(self.right_ids):
            current_state = "OrderError"
            current_message = f"Plates not in descending order"
        else:
            current_state = "Correct"
            current_message = "Plate Order: Correct"

        self.state_buffer.append(current_state)
        if self.state_buffer.count(current_state) >= 3:
            self.confirmed_state = current_state
            self.message = current_message

        current_weight = (
            sum(self.weight_map.get(cls, 0) for cls in self.left_ids + self.right_ids)
            + self.barbell_weight
        )
        self.weight_buffer.append(current_weight)
        try:
            self.total_weight = mode(self.weight_buffer)
        except:
            self.total_weight = current_weight

    def is_safe(self):
        return self.confirmed_state == "Correct"

# === PoseProcessor 類別 ===
class PoseProcessor:
    def __init__(self):
        self.rep_state = "ready"
        self.rep_count = {"Deadlift": 0, "Squat": 0}
        self.phase = "Ready"
        self.start_hip_y = None
        self.start_shoulder_y = None
        self.lowest_hip_y = None
        self.start_rising_y = None
        self.main_person_id = None
        self.locked_action = None

    def detect_action(self, kps, barbell_box):
        if self.locked_action:
            return self.locked_action
        barbell_y = (barbell_box[1] + barbell_box[3]) / 2
        shoulder_top_y = min(kps[5][1], kps[6][1])
        if barbell_y <= shoulder_top_y:
            self.locked_action = "Squat"
        else:
            self.locked_action = "Deadlift"
        return self.locked_action

    def update(self, action, barbell_box, kps):
        if action == "Deadlift":
            self._update_deadlift(barbell_box, kps)
        elif action == "Squat":
            self._update_squat(kps)

    def _update_deadlift(self, barbell_box, kps):
        barbell_y = (barbell_box[1] + barbell_box[3]) / 2
        hip_y = (kps[11][1] + kps[12][1]) / 2
        knee_y = (kps[13][1] + kps[14][1]) / 2
        mid_y = (hip_y + knee_y) / 2

        # 初始化
        if self.start_hip_y is None:
            self.start_hip_y = hip_y
            self.last_barbell_y = barbell_y
            self.phase = "Ready"
            self.trend = "stable"
            self.stable_counter = 0
            return

        # 趨勢判斷（追蹤 barbell_y 的上下）
        threshold = 3
        if barbell_y < self.last_barbell_y - threshold:
            self.trend = "up"
            self.stable_counter = 0
        else:
            self.trend = "stable"
            self.stable_counter += 1

        # 狀態轉移邏輯（類似 squat 方式）
        if self.phase == "Ready":
            if self.trend == "up" and knee_y >= barbell_y > mid_y:
                self.phase = "Up"

        elif self.phase == "Up":
            if self.trend == "up" and mid_y >= barbell_y >= hip_y:
                self.phase = "Finish"
                self.rep_count["Deadlift"] += 1
                self.stable_counter = 0

        elif self.phase == "Finish":
            if barbell_y > knee_y:
                self.phase = "Ready"

        # 更新 barbell_y
        self.last_barbell_y = barbell_y

    def _update_squat(self, kps):
        hip_y = (kps[11][1] + kps[12][1]) / 2

        # === 初始化 ===
        if self.start_hip_y is None:
            self.start_hip_y = hip_y
            self.last_hip_y = hip_y
            self.phase = "Ready"
            self.trend = "stable"
            self.prev_phase = "Ready"
            self.stable_counter = 0
            self.rep_count["Squat"] = 0
            return

        # === 趨勢判斷 ===
        threshold = 3
        if hip_y > self.last_hip_y + threshold:
            self.trend = "down"
            self.stable_counter = 0
        elif hip_y < self.last_hip_y - threshold:
            self.trend = "up"
            self.stable_counter = 0
        else:
            self.trend = "stable"
            self.stable_counter += 1

        # === 狀態流程 ===
        if self.phase == "Ready":
            if hip_y > self.start_hip_y * 1.1 and self.trend == "down":
                self.phase = "Process"

        elif self.phase == "Process":
            if self.trend == "up" and abs(hip_y - self.start_hip_y) <= 10:
                self.phase = "Finish"
                self.rep_count["Squat"] += 1
                self.stable_counter = 0

        elif self.phase == "Finish":
            # 當上一幀不是 Finish → 表示剛進入 Finish，下一幀立即轉 Ready
            if self.stable_counter >= 2:
                self.phase = "Ready"

        # === 更新 ===
        self.last_hip_y = hip_y
        self.prev_phase = self.phase

# === VideoInferenceManager 管理推論 ===
class VideoInferenceManager:
    def __init__(self, pose_model, plate_model, barbell_model, weight_map, class_names, conf=0.7):
        self.conf_threshold = conf
        self.pose_model = pose_model
        self.plate_model = plate_model
        self.barbell_model = barbell_model
        self.class_names = class_names
        self.pose_processor = PoseProcessor()
        self.plate_processor = PlateProcessor(weight_map)
        self.main_center = None

    def _draw_text(self, frame, text, position, color=(0, 255, 0), scale=0.8, thickness=2):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _draw_skeleton(self, frame, kps, action):
        if len(kps) < 17:
            return
        skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for i, j in skeleton:
            pt1, pt2 = tuple(np.int32(kps[i])), tuple(np.int32(kps[j]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        red_lines = []
        if action == "Deadlift":
            red_lines = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        elif action == "Squat":
            red_lines = [(5, 6), (11, 12)]

        for i, j in red_lines:
            if i >= len(kps) or j >= len(kps):
                continue
            pt1, pt2 = tuple(np.int32(kps[i])), tuple(np.int32(kps[j]))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 4)

    def _get_main_person(self, boxes, frame_width):
        cx = frame_width // 2
        return np.argmin([abs(((box[0] + box[2]) // 2) - cx) for box in boxes])

    def _get_barbell_box(self, frame):
        results = self.barbell_model.predict(frame, conf=0.5, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return (x1 + x2) // 2, (x1, y1, x2, y2)
        return None, (0, 0, 0, 0)
    
    def _reset_state(self):
        self.pose_processor = PoseProcessor()
        self.plate_processor = PlateProcessor(self.plate_processor.weight_map)
        self.main_center = None

    def _process_frame(self, frame, frame_width):
        barbell_cx, barbell_box = self._get_barbell_box(frame.copy())
        if barbell_cx is None:
            return frame, ["Warning: Barbell not detected"]
        
        pose = self.pose_model.predict(frame, conf=0.5, verbose=False)[0]
        if not pose.keypoints or pose.keypoints.xy is None or len(pose.keypoints.xy) == 0:
            return frame, ["Warning: Person not detected"]

        boxes = pose.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return frame, ["Warning: No person boxes"]

        # ✅ 改進版主角追蹤，讓 main_center 可隨主角微調位置避免失準
        if self.main_center is None:
            main_id = self._get_main_person(boxes, frame_width)
        else:
            distances = [
                np.sqrt((int((b[0]+b[2])/2)-self.main_center[0])**2 +
                        (int((b[1]+b[3])/2)-self.main_center[1])**2)
                for b in boxes
            ]
            main_id = int(np.argmin(distances))

        box = boxes[main_id]
        self.main_center = (
            int((box[0] + box[2]) / 2),
            int((box[1] + box[3]) / 2)
        )

        if main_id >= len(pose.keypoints.xy):
            return frame, ["Warning: Keypoints missing"]

        kps = pose.keypoints.xy[main_id].cpu().numpy()
        if len(kps) < 17:
            return frame, ["Warning: Incomplete skeleton"]

        box = boxes[main_id].astype(int)
        self._draw_text(frame, "Main Person", (box[0], box[1] - 10), (255, 0, 0))
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        self._draw_text(frame, "Barbell", (barbell_box[0], barbell_box[1] - 10), (0, 153, 255))
        cv2.rectangle(frame, (barbell_box[0], barbell_box[1]), (barbell_box[2], barbell_box[3]), (0, 153, 255), 2)

        action = self.pose_processor.detect_action(kps, barbell_box)
        self.pose_processor.update(action, barbell_box, kps)
        self._draw_skeleton(frame, kps, action)

        plate_results = self.plate_model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        plates = []
        for box in plate_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 >= barbell_box[0] and x2 <= barbell_box[2] and y1 >= barbell_box[1] and y2 <= barbell_box[3]:
                cls = int(box.cls[0])
                cx = (x1 + x2) // 2
                plates.append((cx, cls))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                self._draw_text(frame, self.class_names[cls], (x1, y1 - 10), (0, 255, 255))

        self.plate_processor.update(plates, barbell_box)
        
        self._draw_text(frame, f"Weight: {self.plate_processor.total_weight} kg", (20, 40))
        self._draw_text(frame, self.plate_processor.message, (20, 70), (0, 255, 0) if self.plate_processor.is_safe() else (0, 0, 255))
        self._draw_text(frame, f"Action: {action} {self.pose_processor.phase}", (20, 100))
        self._draw_text(frame, f"Reps: {self.pose_processor.rep_count.get(action, 0)}", (20, 130))
        
        return frame, []

    def _run_on_video(self, input_path, output_path):
        self._reset_state()  # ✅ 每段影片開始前重置狀態
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, warnings = self._process_frame(frame, w)
            for i, msg in enumerate(warnings):
                self._draw_text(processed_frame, msg, (10, 140 + i * 25), (0, 0, 255))

            out.write(processed_frame)

        cap.release()
        out.release()

    def run(self, video_dir, output_dir):
        for file in os.listdir(video_dir):
            if file.lower().endswith((".mp4", ".mov", ".avi")):
                input_path = os.path.join(video_dir, file)
                output_path = os.path.join(output_dir, file)
                self._run_on_video(input_path, output_path)

def run_inference(video_path):
    # 產出影片路徑：output/原檔名
    filename = os.path.basename(video_path)
    output_path = os.path.join("output", filename)

    # 建立推論管理器
    manager = VideoInferenceManager(pose_model, plate_model, barbell_model, WEIGHT_MAPPING, class_names)
    manager._reset_state()
    manager._run_on_video(video_path, output_path)

    # 回傳結果
    action = manager.pose_processor.locked_action or "Unknown"
    reps = manager.pose_processor.rep_count.get(action, 0)
    weight = manager.plate_processor.total_weight
    safe = manager.plate_processor.is_safe()
    warning = None if safe else manager.plate_processor.message

    return {
        "video_path": output_path,
        "action": action,
        "reps": reps,
        "weight": weight,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
