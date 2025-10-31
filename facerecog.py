import sys
import cv2
import numpy as np
import sqlite3
import collections
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# ================== DATABASE ==================
DB_NAME = 'faces.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY,
            name TEXT,
            count INTEGER DEFAULT 0,
            embedding BLOB
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_person(pid, name, embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    emb_bytes = embedding.tobytes()
    cursor.execute('INSERT OR IGNORE INTO persons (id, name, count, embedding) VALUES (?, ?, 0, ?)',
                   (pid, name, emb_bytes))
    conn.commit()
    conn.close()

def increment_visit(pid):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE persons SET count = count + 1 WHERE id = ?', (pid,))
    conn.commit()
    conn.close()

def load_persons():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, count, embedding FROM persons')
    rows = cursor.fetchall()
    conn.close()
    persons = {}
    for row in rows:
        pid, name, count, emb_bytes = row
        embedding = np.frombuffer(emb_bytes, dtype=np.float32) if emb_bytes else None
        persons[pid] = [embedding, name, count]
    return persons

def save_roi(x, y, w, h):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    roi_str = f"{x},{y},{w},{h}"
    cursor.execute('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)', ('roi', roi_str))
    conn.commit()
    conn.close()

def load_roi():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM config WHERE key = ?', ('roi',))
    row = cursor.fetchone()
    conn.close()
    if row:
        x, y, w, h = map(int, row[0].split(','))
        return (x, y, w, h)
    return None

# ================== MOBILENETV2 ==================
print("Loading MobileNetV2...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print("Model loaded.")

def get_embedding(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    x = preprocess_input(resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)
    embedding = model.predict(x, verbose=0)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding.flatten()

# ================== n / l – k RELIABILITY ==================
# ----- 1. New-person vs transient -----
N_FRAMES   = 10          # n – look-back window for new-person detection
L_FAR      =  6          # l – at least L_FAR far embeddings → reject transient
SIM_FAR    = 0.55        # cosine < SIM_FAR → different person
SIM_CLOSE  = 0.75        # cosine ≥ SIM_CLOSE → match known ID

# ----- 2. Visit counting -----
K_CONSECUTIVE = 8        # k – need K consecutive matches to count a visit

recent_embeddings = collections.deque(maxlen=N_FRAMES)   # for new-person detection
consecutive_matches = 0                                 # current streak for a known ID
current_person_id = None                                # ID of the person we are tracking

def is_new_person(new_emb: np.ndarray) -> bool:
    """Return True if the face is a *new* person (not a transient flash)."""
    if len(recent_embeddings) == 0:
        return True
    far_count = 0
    for old in recent_embeddings:
        sim = cosine_similarity([new_emb], [old])[0][0]
        if sim < SIM_FAR:
            far_count += 1
        if far_count >= L_FAR:
            break
    return far_count >= L_FAR

# ================== SHARED DATA ==================
init_db()
known_persons = load_persons()
next_id = max(known_persons.keys(), default=0) + 1
detection_roi = load_roi()

# ================== CAMERA THREAD ==================
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    person_detected_signal = pyqtSignal(int, str, int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def run(self):
        global known_persons, next_id, detection_roi
        global recent_embeddings, consecutive_matches, current_person_id

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # ---- ROI overlay -------------------------------------------------
            if detection_roi:
                x, y, w, h = detection_roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "DETECTION ZONE", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            # ---- pick largest face inside ROI --------------------------------
            largest = None
            max_area = 0
            for (fx, fy, fw, fh) in faces:
                if detection_roi:
                    rx, ry, rw, rh = detection_roi
                    if not (fx >= rx and fy >= ry and fx + fw <= rx + rw and fy + fh <= ry + rh):
                        continue
                area = fw * fh
                if area > max_area:
                    max_area = area
                    largest = (fx, fy, fw, fh)

            if largest:
                x, y, w, h = largest
                face = frame[y:y + h, x:x + w]
                if face.shape[0] < 100 or face.shape[1] < 100:
                    continue

                embedding = get_embedding(face)

                # --------------------------------------------------------------
                # 1. Try to match a *known* person
                # --------------------------------------------------------------
                match_id = None
                best_sim = 0.0
                for pid, info in known_persons.items():
                    if info[0] is not None:
                        sim = cosine_similarity([embedding], [info[0]])[0][0]
                        if sim > best_sim:
                            best_sim = sim
                            match_id = pid

                # --------------------------------------------------------------
                # 2. Decision tree
                # --------------------------------------------------------------
                if match_id and best_sim >= SIM_CLOSE:
                    # ---- KNOWN PERSON ------------------------------------------------
                    display_name = known_persons[match_id][1]

                    # Update recent buffer (helps later new-person detection)
                    recent_embeddings.append(embedding.copy())

                    # ---- Visit counting with k-out-of-n -------------------------------
                    if current_person_id == match_id:
                        consecutive_matches += 1
                    else:
                        # New person just entered – reset streak
                        current_person_id = match_id
                        consecutive_matches = 1

                    if consecutive_matches == K_CONSECUTIVE:
                        # First time we reach the threshold → count the visit
                        known_persons[match_id][2] += 1
                        increment_visit(match_id)
                        self.person_detected_signal.emit(match_id, display_name,
                                                         known_persons[match_id][2])

                else:
                    # ---- POSSIBLY NEW PERSON -----------------------------------------
                    if is_new_person(embedding):
                        # Genuine new person (not a flash)
                        pid = next_id
                        name = f"Person {pid}"
                        known_persons[pid] = [embedding.copy(), name, 0]
                        save_person(pid, name, embedding)
                        next_id += 1

                        # Reset tracking state
                        recent_embeddings.clear()
                        recent_embeddings.append(embedding.copy())
                        current_person_id = pid
                        consecutive_matches = 1   # will need K frames to count

                        self.person_detected_signal.emit(pid, name, 0)
                        display_name = name
                    else:
                        # Transient / flash – ignore completely
                        display_name = "Transient"
                        # Do NOT push into recent_embeddings – we want to forget it

                    # Reset visit-streak because we are not tracking a known ID now
                    current_person_id = None
                    consecutive_matches = 0

                # --------------------------------------------------------------
                # 3. Draw bounding box + info
                # --------------------------------------------------------------
                color = (0, 255, 0) if match_id else (0, 165, 255)  # green = known, orange = new/transient
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                txt = f"{display_name}"
                if match_id:
                    txt += f" ({best_sim:.2f})"
                cv2.putText(frame, txt, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            else:
                # No face → reset tracking
                current_person_id = None
                consecutive_matches = 0

            self.change_pixmap_signal.emit(frame)

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ================== ROI DRAWER ==================
class ROIDrawer(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Draw Detection Area (Click & Drag)")
        self.setFixedSize(640, 480)
        self.parent = parent
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if self.parent.thread and self.parent.thread.cap.isOpened():
            ret, frame = self.parent.thread.cap.read()
            if ret:
                self.frame = frame.copy()
                self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 50 and h > 50:
                orig_w = self.parent.thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                orig_h = self.parent.thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                sx = orig_w / 640
                sy = orig_h / 480
                x, y, w, h = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
                save_roi(x, y, w, h)
                global detection_roi
                detection_roi = (x, y, w, h)
                self.parent.detection_roi = detection_roi
                QMessageBox.information(self, "Success",
                                        f"Detection area updated: {x},{y},{w},{h}")
            self.close()

    def paintEvent(self, event):
        if self.frame is not None:
            painter = QPainter(self)
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qt_img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            scaled = qt_img.scaled(640, 480, Qt.KeepAspectRatio)
            painter.drawImage(0, 0, scaled)
            if self.drawing and self.start_point and self.end_point:
                x1, y1 = self.start_point.x(), self.start_point.y()
                x2, y2 = self.end_point.x(), self.end_point.y()
                painter.setPen(QColor(255, 0, 0, 255))
                painter.setBrush(QColor(255, 0, 0, 50))
                painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

# ================== GUI ==================
class FaceRecognitionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition – n/l-k Visit Counting")
        self.resize(1000, 700)
        self.detection_roi = detection_roi

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Visits"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        self.roi_btn = QPushButton("Set Detection Area")
        self.roi_btn.clicked.connect(self.set_roi)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.table)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.roi_btn)
        self.setLayout(layout)

        self.thread = None
        self.drawer = None
        self.update_table()

    def set_roi(self):
        if not self.thread or not self.thread.cap.isOpened():
            QMessageBox.warning(self, "Error", "Start camera first!")
            return
        self.drawer = ROIDrawer(self)
        self.drawer.show()

    def start_camera(self):
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.person_detected_signal.connect(self.handle_detection)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_image(self, cv_img):
       
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(640, 480, Qt.KeepAspectRatio))

    def handle_detection(self, pid, name, count):
        global known_persons
        if pid in known_persons:
            known_persons[pid][2] = count
        self.update_table()

    def update_table(self):
        global known_persons
        self.table.setRowCount(len(known_persons))
        for i, (pid, info) in enumerate(sorted(known_persons.items())):
            self.table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.table.setItem(i, 1, QTableWidgetItem(info[1]))
            self.table.setItem(i, 2, QTableWidgetItem(str(info[2])))

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

# ================== MAIN ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec_())