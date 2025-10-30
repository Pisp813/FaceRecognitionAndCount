import sys
import cv2
import numpy as np
import sqlite3
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# ================== DATABASE (WITH EMBEDDING) ==================
DB_NAME = 'faces.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY,
            name TEXT,
            count INTEGER DEFAULT 1,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

def save_person(pid, name, embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Convert embedding to bytes
    emb_bytes = embedding.tobytes()
    cursor.execute('INSERT OR IGNORE INTO persons (id, name, count, embedding) VALUES (?, ?, 1, ?)',
                   (pid, name, emb_bytes))
    conn.commit()
    conn.close()

def update_count_db(pid):
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
        if emb_bytes:
            embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        else:
            embedding = None
        persons[pid] = [embedding, name, count]
    return persons

# ================== MOBILENETV2 MODEL ==================
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
    return embedding.flatten()  # 1280-D

# ================== SHARED DATA ==================
init_db()  # Create DB first
known_persons = load_persons()  # Load with embeddings
next_id = max(known_persons.keys(), default=0) + 1
SIMILARITY_THRESHOLD = 0.85

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
        global known_persons, next_id
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            largest = None
            max_area = 0
            for (x, y, w, h) in faces:
                area = w * h
                if area > max_area:
                    max_area = area
                    largest = (x, y, w, h)

            if largest:
                x, y, w, h = largest
                face = frame[y:y+h, x:x+w]
                if face.shape[0] < 100 or face.shape[1] < 100:
                    continue

                embedding = get_embedding(face)

                # Find match
                match_id = None
                best_sim = 0
                for pid, info in known_persons.items():
                    if info[0] is not None:
                        sim = cosine_similarity([embedding], [info[0]])[0][0]
                        if sim > best_sim:
                            best_sim = sim
                            match_id = pid

                if match_id and best_sim > SIMILARITY_THRESHOLD:
                    # Known person
                    known_persons[match_id][2] += 1
                    update_count_db(match_id)
                    self.person_detected_signal.emit(match_id, known_persons[match_id][1], known_persons[match_id][2])
                    display_name = known_persons[match_id][1]
                else:
                    # New person
                    pid = next_id
                    name = f"Person {pid}"
                    known_persons[pid] = [embedding.copy(), name, 1]
                    save_person(pid, name, embedding)  # Save embedding
                    next_id += 1
                    self.person_detected_signal.emit(pid, name, 1)
                    display_name = name

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{display_name} ({best_sim:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.change_pixmap_signal.emit(frame)

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ================== GUI ==================
class FaceRecognitionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition (MobileNetV2 + Embedding in DB)")
        self.resize(1000, 700)

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

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.table)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.thread = None
        self.update_table()

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