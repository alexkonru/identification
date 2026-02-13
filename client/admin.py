import os
import subprocess
import sys
import time
import cv2
import grpc
import PyQt6.QtCore as QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QListWidget, QLabel, QLineEdit, QPushButton,
    QGroupBox, QComboBox, QMessageBox, QInputDialog, QDialog, QFormLayout,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSplitter, QCheckBox, QTableWidget, QTableWidgetItem,
    QScrollArea, QTextEdit, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QIcon
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QMutex

import biometry_pb2
import biometry_pb2_grpc

# --- Dark Theme & Style ---
def set_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

# --- gRPC Client Wrapper ---
class BiometryClient:
    def __init__(self, address='127.0.0.1:50051'):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = biometry_pb2_grpc.GatekeeperStub(self.channel)
        host = address.split(':')[0]
        self.vision_channel = grpc.insecure_channel(f"{host}:50052")
        self.audio_channel = grpc.insecure_channel(f"{host}:50053")
        self.vision_stub = biometry_pb2_grpc.VisionStub(self.vision_channel)
        self.audio_stub = biometry_pb2_grpc.AudioStub(self.audio_channel)

    def wait_until_ready(self, timeout=5.0):
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
            # Real RPC probe: avoids false-positive "channel ready" states
            self.stub.GetSystemStatus(biometry_pb2.Empty(), timeout=timeout)
            return True
        except (grpc.FutureTimeoutError, grpc.RpcError):
            return False

    def list_users(self):
        # Gateway can be up slightly later than UI tabs initialization.
        for _ in range(3):
            try:
                return self.stub.ListUsers(biometry_pb2.ListUsersRequest(), timeout=2.0).users
            except grpc.RpcError:
                time.sleep(0.4)
        return []

    def register_user(self, name, image_bytes):
        return self.stub.RegisterUser(biometry_pb2.RegisterUserRequest(name=name, images=[image_bytes], voices=[]))

    def remove_user(self, user_id):
        return self.stub.RemoveUser(biometry_pb2.IdRequest(id=user_id))

    def list_zones(self):
        return self.stub.ListZones(biometry_pb2.ListZonesRequest()).zones

    def add_zone(self, name):
        return self.stub.AddZone(biometry_pb2.AddZoneRequest(name=name))

    def list_rooms(self):
        return self.stub.ListRooms(biometry_pb2.ListRoomsRequest()).rooms

    def add_room(self, name, zone_id):
        return self.stub.AddRoom(biometry_pb2.AddRoomRequest(name=name, zone_id=zone_id))

    def list_devices(self):
        return self.stub.ListDevices(biometry_pb2.ListDevicesRequest()).devices

    def add_device(self, name, room_id, device_type, connection_string):
        return self.stub.AddDevice(biometry_pb2.AddDeviceRequest(
            name=name, room_id=room_id, device_type=device_type, connection_string=connection_string
        ))

    def set_access_rules(self, user_id, room_ids, zone_ids=None):
        if zone_ids is None:
            zone_ids = []
        return self.stub.SetAccessRules(biometry_pb2.SetAccessRulesRequest(
            user_id=user_id, allowed_room_ids=room_ids, allowed_zone_ids=zone_ids
        ))

    def get_user_access(self, user_id):
        return self.stub.GetUserAccess(biometry_pb2.IdRequest(id=user_id))

    def run_identification_pipeline(self, device_id, image_bytes, audio_bytes=None, sample_rate=16000):
        details = []
        vision_ok = False
        audio_ok = None

        try:
            face = self.vision_stub.ProcessFace(biometry_pb2.ImageFrame(content=image_bytes))
            vision_ok = face.detected and face.is_live
            details.append(
                f"[VISION] detected={face.detected}, live={face.is_live}, "
                f"score={face.liveness_score:.3f}, provider={face.execution_provider or '-'}"
            )
            if face.error_msg:
                details.append(f"[VISION] error={face.error_msg}")
        except grpc.RpcError as e:
            details.append(f"[VISION] RPC ERROR: {e.code().name} {e.details()}")

        if audio_bytes:
            try:
                voice = self.audio_stub.ProcessVoice(
                    biometry_pb2.AudioChunk(content=audio_bytes, sample_rate=sample_rate)
                )
                audio_ok = voice.detected
                details.append(
                    f"[AUDIO] detected={voice.detected}, emb_len={len(voice.embedding)}, "
                    f"provider={voice.execution_provider or '-'}"
                )
                if voice.error_msg:
                    details.append(f"[AUDIO] error={voice.error_msg}")
            except grpc.RpcError as e:
                audio_ok = False
                details.append(f"[AUDIO] RPC ERROR: {e.code().name} {e.details()}")
        else:
            details.append("[AUDIO] skipped")

        try:
            req_kwargs = {"device_id": device_id, "image": image_bytes}
            req_fields = getattr(biometry_pb2.CheckAccessRequest, "DESCRIPTOR", None)
            field_map = req_fields.fields_by_name if req_fields else {}
            if "audio" in field_map:
                req_kwargs["audio"] = audio_bytes or b""
            if "audio_sample_rate" in field_map:
                req_kwargs["audio_sample_rate"] = sample_rate
            access = self.stub.CheckAccess(biometry_pb2.CheckAccessRequest(**req_kwargs))
            stage = getattr(access, "decision_stage", "decision")
            face_live = getattr(access, "face_live", False)
            face_liveness_score = getattr(access, "face_liveness_score", 0.0)
            face_distance = getattr(access, "face_distance", 1.0)
            voice_distance = getattr(access, "voice_distance", 1.0)
            final_confidence = getattr(access, "final_confidence", 0.0)

            details.append(
                f"[DECISION] stage={stage}, granted={access.granted}, user={access.user_name}, msg={access.message}"
            )
            details.append(
                f"[CONF] face_live={face_live}({face_liveness_score:.3f}) "
                f"face_dist={face_distance:.3f}, voice_dist={voice_distance:.3f}, final={final_confidence:.3f}"
            )
            return {
                "ok": True,
                "granted": access.granted,
                "user_name": access.user_name,
                "message": access.message,
                "vision_ok": vision_ok,
                "audio_ok": audio_ok,
                "details": details,
                "stage": stage,
                "face_score": face_liveness_score,
                "face_distance": face_distance,
                "voice_distance": voice_distance,
                "final_confidence": final_confidence,
            }
        except grpc.RpcError as e:
            details.append(f"[GATEWAY] RPC ERROR: {e.code().name} {e.details()}")
            return {
                "ok": False,
                "granted": False,
                "user_name": "-",
                "message": e.details(),
                "vision_ok": vision_ok,
                "audio_ok": audio_ok,
                "details": details,
                "stage": "gateway_error",
                "face_score": 0.0,
                "face_distance": 1.0,
                "voice_distance": 1.0,
                "final_confidence": 0.0,
            }

    def get_system_status(self):
        return self.stub.GetSystemStatus(biometry_pb2.Empty(), timeout=3.0)

    def control_service(self, service, action):
        return self.stub.ControlService(biometry_pb2.ControlServiceRequest(service_name=service, action=action))

    def control_door(self, device_id, command):
        return self.stub.ControlDoor(biometry_pb2.ControlDoorRequest(device_id=device_id, command=command))

    def scan_hardware(self):
        return self.stub.ScanHardware(biometry_pb2.Empty()).found_devices

    def get_logs(self, limit=50, offset=0):
        return self.stub.GetLogs(biometry_pb2.GetLogsRequest(limit=limit, offset=offset)).logs

# --- UI Components ---

class SystemTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(3000)

    def init_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.status_layout = QVBoxLayout(content)
        self.status_widgets = {}
        for svc in ["Gateway", "Database", "Vision", "Audio"]:
            g = QGroupBox(f"{svc} Service")
            l = QFormLayout()
            lbl_status = QLabel("Checking...")
            lbl_device = QLabel("-")
            lbl_msg = QLabel("-")
            lbl_msg.setWordWrap(True)
            l.addRow("Status:", lbl_status)
            l.addRow("Device:", lbl_device)
            l.addRow("Details:", lbl_msg)
            g.setLayout(l)
            self.status_layout.addWidget(g)
            self.status_widgets[svc.lower()] = (lbl_status, lbl_device, lbl_msg)
        scroll.setWidget(content)
        layout.addWidget(scroll, 2)

        svc_group = QGroupBox("Управление службами")
        svc_layout = QFormLayout()
        for name in ["vision-worker", "audio-worker"]:
            hbox = QHBoxLayout()
            btn_start = QPushButton("Start"); btn_start.clicked.connect(lambda _, n=name: self.control_service(n, "start"))
            btn_stop = QPushButton("Stop"); btn_stop.clicked.connect(lambda _, n=name: self.control_service(n, "stop"))
            btn_restart = QPushButton("Restart"); btn_restart.clicked.connect(lambda _, n=name: self.control_service(n, "restart"))
            hbox.addWidget(btn_start); hbox.addWidget(btn_stop); hbox.addWidget(btn_restart)
            svc_layout.addRow(name, hbox)
        svc_group.setLayout(svc_layout)
        layout.addWidget(svc_group, 1)

        hw_group = QGroupBox("Управление оборудованием")
        hw_layout = QVBoxLayout()
        self.btn_scan = QPushButton("🔍 Сканировать сеть")
        self.btn_scan.clicked.connect(self.scan_hardware)
        self.hw_list = QListWidget()
        hw_layout.addWidget(self.btn_scan)
        hw_layout.addWidget(self.hw_list)
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group, 1)

    def refresh_status(self):
        try:
            status = self.client.get_system_status()
            def update(key, s_obj):
                lbl_s, lbl_d, lbl_m = self.status_widgets[key]
                lbl_s.setText("🟢 ONLINE" if s_obj.online else "🔴 OFFLINE")
                lbl_d.setText(f"🖥️ {s_obj.device}")
                lbl_m.setText(s_obj.message)
                if not s_obj.online: lbl_s.setStyleSheet("color: red; font-weight: bold")
                else: lbl_s.setStyleSheet("color: #00ff00; font-weight: bold")
            update("gateway", status.gateway); update("database", status.database); update("vision", status.vision); update("audio", status.audio)
        except Exception as e:
            for k in self.status_widgets:
                self.status_widgets[k][0].setText("🔴 CONNECT ERROR")
                self.status_widgets[k][2].setText(str(e))

    def control_service(self, name, action):
        try: self.client.control_service(name, action); QMessageBox.information(self, "Результат", "Команда отправлена")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

    def scan_hardware(self):
        self.hw_list.clear()
        try:
            devs = self.client.scan_hardware()
            for d in devs: self.hw_list.addItem(f"Found: {d.name} ({d.device_type}) at {d.connection_string}")
        except Exception as e: QMessageBox.critical(self, "Ошибка", str(e))

class PersonnelTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Сотрудники</b>"))
        self.user_list = QListWidget()
        left.addWidget(self.user_list)
        self.btn_delete = QPushButton("🗑 Удалить")
        self.btn_delete.clicked.connect(self.delete_user)
        left.addWidget(self.btn_delete)
        layout.addLayout(left, 1)

        right = QGroupBox("Регистрация")
        rl = QVBoxLayout()
        self.name_input = QLineEdit(); self.name_input.setPlaceholderText("ФИО")
        rl.addWidget(self.name_input)
        self.video_label = QLabel("Камера..."); self.video_label.setMinimumSize(320, 240); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setStyleSheet("background-color: #111;")
        rl.addWidget(self.video_label)
        self.btn_capture = QPushButton("📸 Создать"); self.btn_capture.clicked.connect(self.register_user)
        rl.addWidget(self.btn_capture)
        right.setLayout(rl)
        layout.addWidget(right, 2)
        self.refresh_users()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap: self.cap.release()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def refresh_users(self):
        self.user_list.clear()
        for u in self.client.list_users(): self.user_list.addItem(f"{u.id}: {u.name}")

    def register_user(self):
        name = self.name_input.text()
        if name and self.current_frame is not None:
            _, enc = cv2.imencode('.jpg', self.current_frame)
            try: self.client.register_user(name, enc.tobytes()); self.refresh_users(); self.name_input.clear()
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def delete_user(self):
        item = self.user_list.currentItem()
        if item:
            uid = int(item.text().split(':')[0])
            self.client.remove_user(uid); self.refresh_users()

class InfrastructureTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.setup_ui()
    def setup_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        tb = QHBoxLayout()
        b1 = QPushButton("+ Зона"); b1.clicked.connect(self.add_zone); tb.addWidget(b1)
        b2 = QPushButton("+ Комната"); b2.clicked.connect(self.add_room); tb.addWidget(b2)
        b3 = QPushButton("+ Устройство"); b3.clicked.connect(self.add_device); tb.addWidget(b3)
        left.addLayout(tb)
        
        et = QHBoxLayout()
        self.btn_del = QPushButton("🗑 Удалить"); self.btn_del.setStyleSheet("background-color: #550000"); self.btn_del.clicked.connect(self.delete_item); self.btn_del.setEnabled(False)
        et.addStretch(); et.addWidget(self.btn_del)
        left.addLayout(et)
        
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Инфраструктура", "ID", "Статус"])
        self.tree.currentItemChanged.connect(self.on_select)
        left.addWidget(self.tree)
        layout.addLayout(left, 2)
        
        right = QGroupBox("Управление"); rl = QVBoxLayout()
        self.lbl_info = QLabel("Выберите элемент..."); self.lbl_info.setWordWrap(True)
        rl.addWidget(self.lbl_info)
        self.door_controls = QWidget(); dl = QVBoxLayout(self.door_controls)
        b_o = QPushButton("🔓 Открыть"); b_o.clicked.connect(lambda: self.control_door("open_once"))
        dl.addWidget(QLabel("<b>Замок:</b>")); dl.addWidget(b_o)
        self.door_controls.setVisible(False)
        rl.addWidget(self.door_controls); rl.addStretch()
        right.setLayout(rl)
        layout.addWidget(right, 1)
        self.refresh_tree()

    def refresh_tree(self):
        self.tree.clear()
        try:
            zones = self.client.list_zones(); rooms = self.client.list_rooms(); devices = self.client.list_devices()
            z_map = {z.id: QTreeWidgetItem([z.name, str(z.id), "Zone"]) for z in zones}
            r_map = {r.id: QTreeWidgetItem([r.name, str(r.id), "Room"]) for r in rooms}
            for z in z_map.values(): 
                z.setData(0, Qt.ItemDataRole.UserRole, ("zone", 0))
                self.tree.addTopLevelItem(z)
            for r in rooms:
                if r.zone_id in z_map:
                    item = r_map[r.id]; item.setData(0, Qt.ItemDataRole.UserRole, ("room", r.id))
                    z_map[r.zone_id].addChild(item)
            for d in devices:
                if d.room_id in r_map:
                    item = QTreeWidgetItem([d.name, str(d.id), d.device_type]); item.setData(0, Qt.ItemDataRole.UserRole, ("device", d))
                    r_map[d.room_id].addChild(item)
            self.tree.expandAll()
        except: pass

    def add_zone(self):
        n, ok = QInputDialog.getText(self, "New Zone", "Name:")
        if ok and n: self.client.add_zone(n); self.refresh_tree()
    def add_room(self):
        zones = self.client.list_zones()
        d = QDialog(self); l = QFormLayout(d); ne = QLineEdit(); zc = QComboBox()
        for z in zones: zc.addItem(z.name, z.id)
        l.addRow("Name", ne); l.addRow("Zone", zc); b = QPushButton("Add"); b.clicked.connect(d.accept); l.addRow(b)
        if d.exec() and ne.text(): self.client.add_room(ne.text(), zc.currentData()); self.refresh_tree()
    def add_device(self):
        rooms = self.client.list_rooms()
        d = QDialog(self); l = QFormLayout(d); ne = QLineEdit(); rc = QComboBox()
        for r in rooms: rc.addItem(r.name, r.id)
        tc = QComboBox(); tc.addItems(["lock", "camera"]); ce = QLineEdit()
        l.addRow("Name", ne); l.addRow("Room", rc); l.addRow("Type", tc); l.addRow("Conn", ce)
        b = QPushButton("Add"); b.clicked.connect(d.accept); l.addRow(b)
        if d.exec() and ne.text(): self.client.add_device(ne.text(), rc.currentData(), tc.currentText(), ce.text()); self.refresh_tree()
    def delete_item(self):
        item = self.tree.currentItem()
        if not item: return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "device":
            self.client.stub.RemoveDevice(biometry_pb2.IdRequest(id=data[1].id)); self.refresh_tree()
    def on_select(self, current, prev):
        self.btn_del.setEnabled(False); self.door_controls.setVisible(False)
        if not current: return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "device":
            self.btn_del.setEnabled(True); self.current_dev_id = data[1].id
            self.lbl_info.setText(f"Device: {data[1].name}\nConn: {data[1].connection_string}")
            if data[1].device_type == "lock": self.door_controls.setVisible(True)
    def control_door(self, cmd):
        if hasattr(self, 'current_dev_id'): self.client.control_door(self.current_dev_id, cmd)

class AccessTab(QWidget):
    def __init__(self, client):
        super().__init__(); self.client = client; self.setup_ui()
    def setup_ui(self):
        layout = QHBoxLayout(self); left = QVBoxLayout(); left.addWidget(QLabel("1. Выберите пользователя"))
        self.user_list = QListWidget(); self.user_list.currentItemChanged.connect(self.load_user_rights); left.addWidget(self.user_list); layout.addLayout(left, 1)
        right = QVBoxLayout(); right.addWidget(QLabel("2. Отметьте доступные помещения"))
        self.rights_tree = QTreeWidget(); self.rights_tree.setHeaderLabels(["Помещение"]); right.addWidget(self.rights_tree)
        self.btn_save = QPushButton("💾 Сохранить права"); self.btn_save.clicked.connect(self.save_rights); right.addWidget(self.btn_save); layout.addLayout(right, 2)
        self.refresh()
    def refresh(self):
        self.user_list.clear(); 
        for u in self.client.list_users(): self.user_list.addItem(f"{u.id}: {u.name}")
    def load_user_rights(self, current, prev):
        if not current: return
        self.rights_tree.clear(); uid = int(current.text().split(':')[0]); allowed = set(self.client.get_user_access(uid).allowed_room_ids)
        zones = self.client.list_zones(); rooms = self.client.list_rooms(); z_map = {z.id: QTreeWidgetItem([z.name]) for z in zones}
        for r in rooms:
            item = QTreeWidgetItem([r.name]); item.setCheckState(0, Qt.CheckState.Checked if r.id in allowed else Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, r.id)
            if r.zone_id in z_map: z_map[r.zone_id].addChild(item)
        for z in z_map.values(): self.rights_tree.addTopLevelItem(z); z.setExpanded(True)
    def save_rights(self):
        item = self.user_list.currentItem()
        if not item: return
        uid = int(item.text().split(':')[0]); allowed = []
        it = QTreeWidgetItemIterator(self.rights_tree)
        while it.value():
            if it.value().checkState(0) == Qt.CheckState.Checked and it.value().childCount() == 0:
                allowed.append(it.value().data(0, Qt.ItemDataRole.UserRole))
            it += 1
        self.client.set_access_rules(uid, allowed); QMessageBox.information(self, "OK", "Saved")

class LogTab(QWidget):
    def __init__(self, client):
        super().__init__(); self.client = client; layout = QVBoxLayout(self)
        self.btn_refresh = QPushButton("🔄 Обновить журнал"); self.btn_refresh.clicked.connect(self.load_logs); layout.addWidget(self.btn_refresh)
        self.table = QTableWidget(); self.table.setColumnCount(6); self.table.setHorizontalHeaderLabels(["ID", "Время", "Пользователь", "Помещение", "Доступ", "Детали"]); layout.addWidget(self.table); self.load_logs()
    def load_logs(self):
        self.table.setRowCount(0)
        try:
            logs = self.client.get_logs(limit=100); self.table.setRowCount(len(logs))
            for i, log in enumerate(logs):
                self.table.setItem(i, 0, QTableWidgetItem(str(log.id))); self.table.setItem(i, 1, QTableWidgetItem(log.timestamp))
                self.table.setItem(i, 2, QTableWidgetItem(log.user_name)); self.table.setItem(i, 3, QTableWidgetItem(log.room_name))
                self.table.setItem(i, 4, QTableWidgetItem("✅" if log.access_granted else "❌"))
                self.table.setItem(i, 5, QTableWidgetItem(log.details))
        except: pass

class HelpTab(QWidget):
    def __init__(self):
        super().__init__(); layout = QVBoxLayout(self); text = QTextEdit(); text.setReadOnly(True)
        text.setFont(QFont("Arial", 14))
        text.setHtml("""
        <style>
            h1 { color: #42a5f5; font-size: 28pt; margin-bottom: 20px; }
            h3 { color: #90caf9; font-size: 22pt; margin-top: 30px; margin-bottom: 10px; }
            p, li { font-size: 16pt; line-height: 1.6; }
            b { color: #e1f5fe; }
            .code { font-family: monospace; background-color: #333; padding: 5px; border-radius: 4px; font-size: 14pt; }
        </style>
        <h1>📘 Справка по системе Biometry Admin</h1>
        
        <h3>🔌 Быстрый старт</h3>
        <p>1. Запустите серверную часть: <span class=\"code\">./start_all.sh</span></p>
        <p>2. Убедитесь, что статусы на вкладке <b>"Система"</b> зеленые (ONLINE).</p>

        <h3>🧠 Мониторинг и Идентификация</h3>
        <p>Вкладка <b>"📹 Мониторинг"</b> позволяет видеть работу системы в реальном времени.</p>
        <p><b>Режимы работы:</b></p>
        <ul>
            <li><b>Пассивный режим (по умолчанию):</b> Клиент просто показывает видео. Распознавание выполняет отдельный <i>Hardware Controller</i> (если запущен). Результаты подтягиваются из журнала.</li>
            <li><b>Активный режим (Live ID):</b> Включите галочку <b>"🚀 Live ID"</b>. Клиент будет сам отправлять кадры на сервер. Идеально для тестов без контроллера.</li>
        </ul>
        <p><b>Расшифровка данных на экране:</b></p>
        <ul>
            <li><b>Liveness:</b> Вероятность того, что лицо живое (не фото). Порог > 50%.</li>
            <li><b>Distance:</b> Степень отличия лица от эталона. Чем меньше, тем лучше (Порог < 0.6).</li>
            <li><b>Provider:</b> Где выполняются вычисления (CUDA - видеокарта, CPU - процессор).</li>
        </ul>

        <h3>🏗 Настройка</h3>
        <p>1. Создайте <b>Зону</b> -> <b>Комнату</b> -> <b>Камеру</b>.</p>
        <p>2. Для веб-камеры используйте Connection: <span class="code">0</span>.</p>
        <p>3. Назначьте права сотрудникам на вкладке <b>"Доступ"</b>.</p>
        """)
        layout.addWidget(text)

class MonitoringTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.active_cameras = {}
        self.selected_device_id = None
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.process_active_mode)
        self.mic_stream = MicrophoneStream(sample_rate=16000, channels=1)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self.on_select)
        left.addWidget(QLabel("<b>Помещения и устройства:</b>"))
        left.addWidget(self.tree)

        self.btn_refresh = QPushButton("🔄 Обновить список инфраструктуры")
        self.btn_refresh.clicked.connect(self.refresh_tree)
        left.addWidget(self.btn_refresh)

        self.btn_reconnect = QPushButton("🔌 Переподключить выбранную камеру")
        self.btn_reconnect.clicked.connect(self.reconnect_selected_camera)
        left.addWidget(self.btn_reconnect)
        left.addStretch()
        layout.addLayout(left, 1)

        center = QVBoxLayout()
        self.video_label = QLabel("Выбери устройство-камеру слева")
        self.video_label.setMinimumSize(800, 600)  # 4:3 area
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #555;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.video_label)

        layout.addLayout(center, 3)

        self.pipeline_log = QTextEdit()
        self.pipeline_log.setReadOnly(True)
        self.pipeline_log.setMinimumWidth(420)
        self.pipeline_log.setPlaceholderText("Здесь отображается полный пайплайн: liveness -> face -> voice -> policy -> decision")
        layout.addWidget(self.pipeline_log, 2)

        self.refresh_tree()
        self.log_timer.start(1200)  # always-on identification

    def capture_audio_raw(self, duration_s=1, sample_rate=16000):
        chunk = self.mic_stream.read_chunk(duration_s=duration_s)
        if chunk is None:
            self.pipeline_log.append("[AUDIO] capture failed: microphone stream unavailable")
            return None
        return chunk

    def refresh_tree(self):
        self.tree.clear()
        try:
            zones = self.client.list_zones()
            rooms = self.client.list_rooms()
            devices = self.client.list_devices()
            z_map = {z.id: QTreeWidgetItem([f"{z.name}"]) for z in zones}
            r_map = {}
            for r in rooms:
                room_item = QTreeWidgetItem([f"{r.name}"])
                room_item.setData(0, Qt.ItemDataRole.UserRole, ("room", r.id))
                r_map[r.id] = room_item
                if r.zone_id in z_map:
                    z_map[r.zone_id].addChild(room_item)

            for d in devices:
                if d.room_id in r_map and d.device_type == 'camera':
                    d_item = QTreeWidgetItem([f"📷 {d.name} ({d.connection_string})"])
                    d_item.setData(0, Qt.ItemDataRole.UserRole, ("device", d))
                    r_map[d.room_id].addChild(d_item)

            for z in z_map.values():
                self.tree.addTopLevelItem(z)
                z.setExpanded(True)
        except Exception as e:
            self.pipeline_log.append(f"[UI] refresh error: {e}")

    def on_select(self, current, _prev=None):
        self.stop_all_videos()
        if not current:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "device":
            self.selected_device_id = None
            self.video_label.setText("Выбери устройство-камеру слева")
            return

        dev = data[1]
        self.selected_device_id = dev.id
        t = VideoThread(dev.connection_string, dev.id)
        t.frame_ready.connect(self._show_frame)
        t.status_msg.connect(lambda msg: self.pipeline_log.append(f"[CAMERA] {msg}"))
        t.start()
        self.active_cameras[dev.id] = t

    def _show_frame(self, img):
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def reconnect_selected_camera(self):
        item = self.tree.currentItem()
        if item:
            self.on_select(item)

    def stop_all_videos(self):
        for t in self.active_cameras.values():
            t.stop()
            t.wait()
        self.active_cameras.clear()

    def restart_videos(self):
        item = self.tree.currentItem()
        if item:
            self.on_select(item)

    def process_active_mode(self):
        # Always active: always use mic + camera for pipeline
        if not self.active_cameras:
            return

        audio_bytes = self.capture_audio_raw(duration_s=1, sample_rate=16000)

        for dev_id, t in self.active_cameras.items():
            frame = t.get_frame_bytes()
            if not frame:
                continue
            try:
                result = self.client.run_identification_pipeline(dev_id, frame, audio_bytes=audio_bytes)
                msg = f"{result['user_name']}: {'OK' if result['granted'] else 'NO'}\n{result['message']}"
                color = (0, 255, 0) if result['granted'] else (0, 0, 255)
                t.set_overlay(msg, color)

                self.pipeline_log.clear()
                self.pipeline_log.append(f"Device ID: {dev_id}")
                self.pipeline_log.append("=" * 50)
                for line in result["details"]:
                    self.pipeline_log.append(line)
                self.pipeline_log.append("=" * 50)
                self.pipeline_log.append(
                    f"RESULT: {'GRANTED' if result['granted'] else 'DENIED'} | STAGE={result['stage']} | "
                    f"VISION_OK={result['vision_ok']} | AUDIO_OK={result['audio_ok']} | CONF={result['final_confidence']:.3f}"
                )
            except Exception as e:
                self.pipeline_log.append(f"[PIPELINE] ERROR: {e}")

    def closeEvent(self, event):
        self.stop_all_videos()
        self.mic_stream.stop()
        super().closeEvent(event)


class MicrophoneStream:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.proc = None

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        cmd = [
            "arecord", "-q", "-r", str(self.sample_rate), "-c", str(self.channels),
            "-f", "S16_LE", "-t", "raw", "-",
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception:
            self.proc = None
            return False

    def read_chunk(self, duration_s=1):
        if not self.start() or not self.proc or not self.proc.stdout:
            return None
        needed = int(self.sample_rate * duration_s * 2 * self.channels)  # int16 mono
        data = self.proc.stdout.read(needed)
        if not data or len(data) < needed:
            return None
        return data

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except Exception:
                self.proc.kill()
        self.proc = None


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)
    status_msg = pyqtSignal(str)

    def __init__(self, source, dev_id):
        super().__init__()
        self.source = int(source) if str(source).isdigit() else source
        self.dev_id = dev_id
        self.running = True
        self.overlay = ("", (0, 255, 0), 0)
        self.mutex = QMutex()
        self.current_frame = None

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_msg.emit(f"Не удалось открыть камеру: source={self.source}")
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.mutex.unlock()
                if self.overlay[2] > 0:
                    y0, dy = 50, 30
                    for i, line in enumerate(self.overlay[0].split('\n')):
                        cv2.putText(frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.overlay[1], 2)
                    self.overlay = (self.overlay[0], self.overlay[1], self.overlay[2] - 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_ready.emit(img)
            else:
                self.status_msg.emit("Поток камеры недоступен")
                self.msleep(200)
            self.msleep(30)
        cap.release()

    def stop(self):
        self.running = False

    def set_overlay(self, text, color):
        self.overlay = (text, color, 60)

    def get_frame_bytes(self):
        self.mutex.lock()
        f = self.current_frame
        self.mutex.unlock()
        if f is not None:
            _, enc = cv2.imencode('.jpg', f)
            return enc.tobytes()
        return None


class AdminApp(QMainWindow):
    def __init__(self, client):
        super().__init__(); self.setWindowTitle("Biometry Admin Panel 2.0"); self.resize(1200, 800); self.client = client
        self.tabs = QTabWidget(); self.personnel_tab = PersonnelTab(self.client); self.monitor_tab = MonitoringTab(self.client)
        self.tabs.addTab(SystemTab(self.client), "⚙️ Система"); self.tabs.addTab(self.personnel_tab, "👥 Персонал")
        self.tabs.addTab(InfrastructureTab(self.client), "🏗 Инфраструктура"); self.tabs.addTab(AccessTab(self.client), "🔐 Доступ")
        self.tabs.addTab(self.monitor_tab, "📹 Мониторинг"); self.tabs.addTab(LogTab(self.client), "📜 Журнал"); self.tabs.addTab(HelpTab(), "❓ Справка")
        self.tabs.currentChanged.connect(self.on_tab_change); self.setCentralWidget(self.tabs)
    def on_tab_change(self, index):
        if self.tabs.widget(index) == self.personnel_tab: self.personnel_tab.start_camera()
        else: self.personnel_tab.stop_camera()
        if self.tabs.widget(index) == self.monitor_tab: self.monitor_tab.restart_videos()
        else: self.monitor_tab.stop_all_videos()
    def closeEvent(self, event):
        self.personnel_tab.stop_camera(); self.monitor_tab.stop_all_videos(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)

    gateway_addr = os.getenv("GATEWAY_ADDR", "127.0.0.1:50051")
    if len(sys.argv) > 1 and sys.argv[1].strip():
        gateway_addr = sys.argv[1].strip()

    client = BiometryClient(gateway_addr)
    if not client.wait_until_ready(timeout=6.0):
        QMessageBox.critical(
            None,
            "Gateway недоступен",
            f"Не удалось подключиться к gRPC Gateway по адресу {gateway_addr}.\n"
            "Проверь, что gateway-service запущен (например, ./start_all.sh или docker compose up).",
        )
        sys.exit(1)

    w = AdminApp(client)
    w.show()
    sys.exit(app.exec())
