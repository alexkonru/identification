import os
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
import cv2
import grpc
import numpy as np
import PyQt6.QtCore as QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QListWidget, QListWidgetItem, QLabel, QLineEdit, QPushButton,
    QGroupBox, QComboBox, QMessageBox, QInputDialog, QDialog, QFormLayout,
    QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator, QHeaderView, QSplitter, QTableWidget, QTableWidgetItem,
    QSizePolicy,
    QTextEdit, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QFont
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex

import biometry_pb2
import biometry_pb2_grpc


PHI = 1.61803398875
INV_PHI = 1.0 / PHI
DEFAULT_UI_SETTINGS = {"theme": "light", "font_size": 13, "voice_modality": True, "show_system_log": False}
SYSTEM_MESSAGES_MAX_LINES = 160
PIPELINE_LOG_MAX_LINES = 80
SYSTEM_LOG_MAX_LINES = 120


def parse_kv_env_file(path: Path) -> dict:
    cfg = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg

# --- Тема и настройки UI ---
UI_SETTINGS_PATH = Path.home() / ".biometry_ui_settings.json"
PIPELINE_RUN_FLAG_PATH = Path(__file__).resolve().parents[1] / ".system_run"
HELP_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "docs" / "admin_help.md"
USER_GUIDE_PATH = Path(__file__).resolve().parents[1] / "docs" / "user_guide.md"


def load_ui_settings() -> dict:
    if not UI_SETTINGS_PATH.exists():
        return DEFAULT_UI_SETTINGS.copy()
    try:
        return json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_UI_SETTINGS.copy()


def save_ui_settings(data: dict):
    UI_SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pipeline_run_flag():
    try:
        if not PIPELINE_RUN_FLAG_PATH.exists():
            return None
        raw = PIPELINE_RUN_FLAG_PATH.read_text(encoding="utf-8").strip()
        if raw == "1":
            return True
        if raw == "0":
            return False
    except Exception:
        return None
    return None


def set_light_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(242, 245, 250))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(20, 26, 38))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 239, 248))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 26, 38))
    palette.setColor(QPalette.ColorRole.Text, QColor(20, 26, 38))
    palette.setColor(QPalette.ColorRole.Button, QColor(233, 239, 248))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(20, 26, 38))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(166, 42, 38))
    palette.setColor(QPalette.ColorRole.Link, QColor(15, 90, 158))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(15, 90, 158))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def set_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 34, 42))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(235, 240, 247))
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 22, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 40, 50))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(35, 40, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(235, 240, 247))
    palette.setColor(QPalette.ColorRole.Text, QColor(235, 240, 247))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 52, 66))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(235, 240, 247))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 100, 90))
    palette.setColor(QPalette.ColorRole.Link, QColor(95, 176, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(46, 129, 214))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def apply_ui_theme(app, settings: dict):
    theme = settings.get("theme", DEFAULT_UI_SETTINGS["theme"])
    if theme == "dark":
        set_dark_theme(app)
        app.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #4e5b74; background: #1f2530; }
            QTabBar::tab { padding: 8px 14px; background: #333c4b; border: 1px solid #4e5b74; border-bottom: none; margin-right: 2px; color: #d9e2ef; }
            QTabBar::tab:selected { background: #1f2530; color: #8ed0ff; font-weight: 600; }
            QGroupBox { border: 1px solid #4e5b74; border-radius: 8px; margin-top: 8px; padding: 10px; background: #262d39; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8ed0ff; font-weight: 600; }
            QPushButton { border: 1px solid #5b6b89; border-radius: 7px; padding: 6px 10px; background: #3a465a; color: #e8eff9; }
            QPushButton:hover { background: #44526a; }
            QPushButton:pressed { background: #314055; }
            QPushButton:disabled { color: #9aa5b9; background: #2e3644; border-color: #3e485b; }
            QLineEdit, QComboBox, QTextEdit, QListWidget, QTreeWidget, QTableWidget { border: 1px solid #55617a; border-radius: 6px; background: #1b212c; color: #e8eff9; }
            QLabel#liveOn { color: #7ddd7d; font-weight: 700; }
            QLabel#liveOff { color: #ff8a80; font-weight: 700; }
        """)
    else:
        set_light_theme(app)
        app.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #aebfd8; background: #ffffff; }
            QTabBar::tab { padding: 8px 14px; background: #e7eef8; border: 1px solid #aebfd8; border-bottom: none; margin-right: 2px; }
            QTabBar::tab:selected { background: #ffffff; color: #0d4f90; font-weight: 600; }
            QGroupBox { border: 1px solid #b7c7de; border-radius: 8px; margin-top: 8px; padding: 10px; background: #f9fbff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #0d4f90; font-weight: 600; }
            QPushButton { border: 1px solid #8ea8cc; border-radius: 7px; padding: 6px 10px; background: #e9f0fb; color: #152139; }
            QPushButton:hover { background: #dce9fa; }
            QPushButton:pressed { background: #d1e2f8; }
            QPushButton:disabled { color: #7f8da5; background: #edf1f8; border-color: #c3cede; }
            QLineEdit, QComboBox, QTextEdit, QListWidget, QTreeWidget, QTableWidget { border: 1px solid #aebfd8; border-radius: 6px; background: #ffffff; color: #152139; }
            QLabel#liveOn { color: #1d6f2b; font-weight: 700; }
            QLabel#liveOff { color: #b3362a; font-weight: 700; }
        """)
    font = app.font()
    font.setPointSize(int(settings.get("font_size", DEFAULT_UI_SETTINGS["font_size"])))
    app.setFont(font)
    app.processEvents()


def trim_text_edit_lines(text_edit: QTextEdit, max_lines: int):
    doc = text_edit.document()
    while doc.blockCount() > max_lines:
        cursor = text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        cursor.select(cursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.deleteChar()


def format_iso_to_local(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = QtCore.QDateTime.fromString(ts, Qt.DateFormat.ISODateWithMs)
        if not dt.isValid():
            dt = QtCore.QDateTime.fromString(ts, Qt.DateFormat.ISODate)
        if dt.isValid():
            return dt.toLocalTime().toString("dd.MM.yyyy HH:mm:ss")
    except Exception:
        pass
    return ts

# --- gRPC-клиент ---
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
        self.pipeline_running_cache = True

    def wait_until_ready(self, total_timeout=45.0, probe_timeout=2.0):
        # Для Docker-старта gateway может подняться не мгновенно:
        # контейнер уже существует, но gRPC сервис ещё инициализируется.
        # Важно: GetSystemStatus внутри gateway опрашивает audio/vision workers,
        # поэтому при их старте/недоступности этот RPC может таймаутить,
        # даже если сам gateway уже доступен.
        deadline = time.time() + total_timeout
        while time.time() < deadline:
            try:
                grpc.channel_ready_future(self.channel).result(timeout=probe_timeout)
            except grpc.FutureTimeoutError:
                time.sleep(0.5)
                continue

            try:
                # Быстрая проверка доступности, если RPC уже отвечает.
                self.stub.GetSystemStatus(biometry_pb2.Empty(), timeout=probe_timeout)
                return True
            except grpc.RpcError as e:
                # Gateway может быть уже поднят, но probe может зависеть от worker'ов.
                # В таких кейсах считаем подключение к gateway успешным.
                if e.code() in {
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.UNIMPLEMENTED,
                    grpc.StatusCode.UNKNOWN,
                    grpc.StatusCode.INTERNAL,
                }:
                    return True
                time.sleep(0.5)

        return False


    def list_users(self):
        # Gateway может подняться чуть позже инициализации вкладок UI.
        for _ in range(3):
            try:
                return self.stub.ListUsers(biometry_pb2.ListUsersRequest(), timeout=2.0).users
            except grpc.RpcError:
                time.sleep(0.4)
        return []

    def register_user(self, name, image_bytes_list, voice_bytes_list=None):
        if isinstance(image_bytes_list, (bytes, bytearray)):
            images = [bytes(image_bytes_list)]
        else:
            images = [bytes(x) for x in (image_bytes_list or []) if x]
        if not images:
            raise ValueError("images list is empty")
        if isinstance(voice_bytes_list, (bytes, bytearray)):
            voices = [bytes(voice_bytes_list)]
        else:
            voices = [bytes(x) for x in (voice_bytes_list or []) if x]
        return self.stub.RegisterUser(
            biometry_pb2.RegisterUserRequest(name=name, images=images, voices=voices)
        )

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

    @staticmethod
    def _message_field_map(message_cls):
        req_fields = getattr(message_cls, "DESCRIPTOR", None)
        return req_fields.fields_by_name if req_fields else {}

    def set_access_rules(self, user_id, room_ids, zone_ids=None):
        if zone_ids is None:
            zone_ids = []
        field_map = self._message_field_map(biometry_pb2.SetAccessRulesRequest)
        kwargs = {"user_id": user_id, "allowed_room_ids": room_ids}
        if "allowed_zone_ids" in field_map:
            kwargs["allowed_zone_ids"] = zone_ids
        return self.stub.SetAccessRules(biometry_pb2.SetAccessRulesRequest(**kwargs))

    def get_user_access(self, user_id):
        return self.stub.GetUserAccess(biometry_pb2.IdRequest(id=user_id))


    def get_system_status(self):
        return self.stub.GetSystemStatus(biometry_pb2.Empty(), timeout=3.0)

    def control_service(self, service, action):
        return self.stub.ControlService(biometry_pb2.ControlServiceRequest(service_name=service, action=action))

    def set_pipeline_running(self, enabled: bool):
        action = "start" if enabled else "stop"
        resp = self.control_service("pipeline", action)
        self.pipeline_running_cache = bool(enabled)
        return resp

    def get_pipeline_running(self):
        state = read_pipeline_run_flag()
        if state is not None:
            self.pipeline_running_cache = state
        return self.pipeline_running_cache

    def control_door(self, device_id, command):
        return self.stub.ControlDoor(biometry_pb2.ControlDoorRequest(device_id=device_id, command=command))

    def scan_hardware(self):
        return self.stub.ScanHardware(biometry_pb2.Empty()).found_devices

    def remove_device(self, device_id):
        return self.stub.RemoveDevice(biometry_pb2.IdRequest(id=device_id))

    def apply_runtime_mode(self, mode="auto", restart_services=True):
        return self.stub.ApplyRuntimeMode(
            biometry_pb2.RuntimeModeRequest(mode=mode, restart_services=restart_services)
        )

    def get_logs(self, limit=50, offset=0):
        return self.stub.GetLogs(biometry_pb2.GetLogsRequest(limit=limit, offset=offset)).logs

    def submit_door_observation(self, session_id, device_id, frame, audio=b"", sample_rate=16000, timestamp_ms=0):
        return None

    def check_access(self, device_id, image_bytes, audio_bytes=None, sample_rate=16000):
        field_map = self._message_field_map(biometry_pb2.CheckAccessRequest)
        kwargs = {"device_id": device_id, "image": image_bytes}
        if "audio" in field_map:
            kwargs["audio"] = audio_bytes or b""
        if "audio_sample_rate" in field_map:
            kwargs["audio_sample_rate"] = sample_rate
        return self.stub.CheckAccess(biometry_pb2.CheckAccessRequest(**kwargs))

    def check_access_v2(self, session_id, device_id, frames, audio_bytes=None, sample_rate=16000, frame_timestamps_ms=None):
        if not hasattr(self.stub, "CheckAccessV2"):
            return None
        req_cls = getattr(biometry_pb2, "CheckAccessRequestV2", None)
        if req_cls is None:
            return None
        field_map = self._message_field_map(req_cls)
        kwargs = {
            "session_id": session_id,
            "device_id": device_id,
            "frames": frames,
            "audio": audio_bytes or b"",
            "audio_sample_rate": sample_rate,
        }
        if frame_timestamps_ms is None:
            frame_timestamps_ms = []
        if "frame_timestamps_ms" in field_map:
            kwargs["frame_timestamps_ms"] = frame_timestamps_ms
        return self.stub.CheckAccessV2(req_cls(**kwargs))

    def stream_device_media(self, device_id, target_fps=6, include_audio=True, audio_sample_rate=16000, audio_chunk_ms=220):
        req_cls = getattr(biometry_pb2, "DeviceMediaRequest", None)
        if req_cls is None or not hasattr(self.stub, "StreamDeviceMedia"):
            raise RuntimeError("StreamDeviceMedia is not available in current proto/gateway build")
        req = req_cls(
            device_id=device_id,
            target_fps=target_fps,
            include_audio=include_audio,
            audio_sample_rate=audio_sample_rate,
            audio_chunk_ms=audio_chunk_ms,
        )
        return self.stub.StreamDeviceMedia(req)


# --- Компоненты UI ---

class SystemTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(3000)
        self._last_sys_msg = ""

    def init_ui(self):
        layout = QVBoxLayout(self)

        status_group = QGroupBox("Статус сервисов")
        grid = QGridLayout()
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self.status_widgets = {}
        labels = [
            ("gateway", "Шлюз"),
            ("database", "БД"),
            ("vision", "Лицо"),
            ("audio", "Голос"),
        ]
        hdr1 = QLabel("<b>Сервис</b>")
        hdr2 = QLabel("<b>Статус</b>")
        hdr3 = QLabel("<b>Устройство</b>")
        grid.addWidget(hdr1, 0, 0)
        grid.addWidget(hdr2, 0, 1)
        grid.addWidget(hdr3, 0, 2)
        for i, (key, title) in enumerate(labels, start=1):
            title_lbl = QLabel(title)
            status_lbl = QLabel("⏳")
            device_lbl = QLabel("-")
            grid.addWidget(title_lbl, i, 0)
            grid.addWidget(status_lbl, i, 1)
            grid.addWidget(device_lbl, i, 2)
            self.status_widgets[key] = (status_lbl, device_lbl)
        status_group.setLayout(grid)
        layout.addWidget(status_group, 2)

        msg_group = QGroupBox("Сообщения системы")
        msg_layout = QVBoxLayout()
        self.system_messages = QTextEdit()
        self.system_messages.setReadOnly(True)
        self.system_messages.setPlaceholderText("Сюда выводятся сообщения о функционировании системы")
        msg_layout.addWidget(self.system_messages)
        msg_group.setLayout(msg_layout)
        layout.addWidget(msg_group, 1)

        # Настройка runtime-режима запуска (CPU/GPU).
        rt_group = QGroupBox("Runtime настройки (CPU/GPU)")
        rt_layout = QFormLayout()

        self.lbl_runtime_info = QLabel("Ожидание статуса...")
        self.lbl_runtime_info.setWordWrap(True)
        self.lbl_runtime_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rt_layout.addRow("Текущий runtime:", self.lbl_runtime_info)
        rt_group.setLayout(rt_layout)
        layout.addWidget(rt_group, 1)

    def _update_status_widget(self, key, service_status):
        lbl_status, lbl_device = self.status_widgets[key]
        online = bool(service_status.online)
        lbl_status.setText("🟢 Онлайн" if online else "🔴 Оффлайн")
        lbl_device.setText(service_status.device)
        color = "#1d6f2b" if online else "#b3362a"
        lbl_status.setStyleSheet(f"color: {color}; font-weight: bold")

    def _update_runtime_info(self, status):
        runtime_file = Path(__file__).resolve().parents[1] / ".server_runtime.env"
        runtime_cfg = parse_kv_env_file(runtime_file)

        vision_threads = runtime_cfg.get("VISION_INTRA_THREADS", "-")
        audio_threads = runtime_cfg.get("AUDIO_INTRA_THREADS", "-")
        vision_cpu = runtime_cfg.get("VISION_FORCE_CPU", "?")
        audio_cpu = runtime_cfg.get("AUDIO_FORCE_CPU", "?")
        audio_cuda = runtime_cfg.get("AUDIO_USE_CUDA", "?")

        vision_device = status.vision.device or "неизвестно"
        audio_device = status.audio.device or "неизвестно"
        mode = (
            "GPU"
            if ("CUDA" in vision_device.upper() or "CUDA" in audio_device.upper() or audio_cuda == "1")
            else "CPU"
        )

        self.lbl_runtime_info.setText(
            f"Выбранный режим: {mode}\n"
            f"Модуль видео (Vision): устройство={vision_device}, потоки ONNX={vision_threads}, FORCE_CPU={vision_cpu}\n"
            f"Модуль аудио (Audio): устройство={audio_device}, потоки ONNX={audio_threads}, FORCE_CPU={audio_cpu}, USE_CUDA={audio_cuda}\n"
            f"Статус Vision: {status.vision.message}\n"
            f"Статус Audio: {status.audio.message}"
        )

    @staticmethod
    def _humanize_gateway_message(raw: str) -> str:
        text = (raw or "").strip()
        if not text.startswith("points_ready="):
            return text
        m = re.search(r"points_ready=(\d+)/(\d+)", text)
        if not m:
            return text
        ready = int(m.group(1))
        total = int(m.group(2))
        out = [f"Точки прохода готовы: {ready} из {total}."]
        if "disabled=" not in text:
            return "\n".join(out)

        disabled_part = text.split("disabled=", 1)[1].strip()
        if not disabled_part:
            return "\n".join(out)

        rooms = [p.strip() for p in disabled_part.split(" | ") if p.strip()]
        if not rooms:
            return "\n".join(out)

        out.append("Проблемные точки:")
        for room in rooms:
            # Формат от gateway:
            # "<zone> / <room>: <reason> (cam=0, lock=1, mic=0)"
            mm = re.search(r"^(.*?):\s*(.*?)\s*\(cam=(\d+),\s*lock=(\d+),\s*mic=(\d+)\)", room)
            if not mm:
                out.append(f"• {room}")
                continue
            room_name = mm.group(1).strip()
            reason = mm.group(2).strip()
            cam = int(mm.group(3))
            lock = int(mm.group(4))
            mic = int(mm.group(5))
            reasons = []
            if cam == 0:
                reasons.append("нет камеры")
            elif cam > 1:
                reasons.append(f"камер: {cam} (должна быть 1)")
            if lock == 0:
                reasons.append("нет замка")
            elif lock > 1:
                reasons.append(f"замков: {lock} (должен быть 1)")
            if mic > 1:
                reasons.append(f"микрофонов: {mic} (должен быть максимум 1)")
            if mic == 0:
                reasons.append("микрофон не подключен")
            detail = ", ".join(reasons) if reasons else reason
            out.append(f"• {room_name}: {detail}")
        return "\n".join(out)

    def refresh_status(self):
        try:
            status = self.client.get_system_status()
            gateway_human = self._humanize_gateway_message(getattr(status.gateway, "message", ""))
            self._update_status_widget("gateway", status.gateway)
            self._update_status_widget("database", status.database)
            self._update_status_widget("vision", status.vision)
            self._update_status_widget("audio", status.audio)
            status_msg = (
                f"gateway: {gateway_human}\n"
                f"vision: {status.vision.message}\n"
                f"audio: {status.audio.message}\n"
                f"database: {status.database.message}"
            )
            if status_msg != self._last_sys_msg:
                self._last_sys_msg = status_msg
                ts = time.strftime("%H:%M:%S")
                self.system_messages.append(f"[{ts}] {status_msg}")
                trim_text_edit_lines(self.system_messages, SYSTEM_MESSAGES_MAX_LINES)
            self._update_runtime_info(status)
        except Exception as e:
            for k in self.status_widgets:
                self.status_widgets[k][0].setText("🔴 Ошибка соединения")
                self.status_widgets[k][1].setText(str(e))


class RegistrationCameraDialog(QDialog):
    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Регистрация пользователя")
        self.resize(900, 620)
        self.current_frame = None
        self.current_audio = b""
        self.required_shots = 5
        self.captured_images = []
        self.required_voice_samples = 2
        self.captured_voices = []
        self.stream_thread = None
        self.camera_devices = []
        self.selected_device_id = None

        layout = QVBoxLayout(self)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("ФИО сотрудника")
        layout.addWidget(self.name_input)
        self.camera_combo = QComboBox()
        layout.addWidget(self.camera_combo)
        self.progress_label = QLabel(f"Снимки: 0/{self.required_shots}. Сделайте 5 фото под разными ракурсами.")
        layout.addWidget(self.progress_label)
        self.voice_progress_label = QLabel(
            f"Голос: 0/{self.required_voice_samples}. Запишите короткую фразу для биометрии."
        )
        layout.addWidget(self.voice_progress_label)

        self.video_label = QLabel("Подключение камеры...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(760, 480)
        self.video_label.setStyleSheet("background:#111;border:1px solid #444;")
        layout.addWidget(self.video_label)

        btns = QHBoxLayout()
        self.btn_capture = QPushButton(f"📸 Сделать снимок (1/{self.required_shots})")
        self.btn_capture.clicked.connect(self.register_user)
        self.btn_record_voice = QPushButton(f"🎤 Записать голос (1/{self.required_voice_samples})")
        self.btn_record_voice.clicked.connect(self.capture_voice_sample)
        btns.addStretch()
        btns.addWidget(self.btn_record_voice)
        btns.addWidget(self.btn_capture)
        btns.addStretch()
        layout.addLayout(btns)
        self.reload_cameras()

    def reload_cameras(self):
        self.camera_devices = [d for d in self.client.list_devices() if d.device_type == "camera"]
        self.camera_combo.clear()
        for d in self.camera_devices:
            self.camera_combo.addItem(f"{d.name} (id={d.id})", d.id)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        if self.camera_devices:
            self.on_camera_changed(0)
        else:
            self.video_label.setText("Нет камер в инфраструктуре")
            self.btn_capture.setEnabled(False)
            self.btn_record_voice.setEnabled(False)

    def on_camera_changed(self, index):
        if index < 0 or index >= len(self.camera_devices):
            return
        self.selected_device_id = int(self.camera_combo.itemData(index))
        self.start_stream_for_selected()

    def start_stream_for_selected(self):
        self.stop_stream()
        if self.selected_device_id is None:
            return
        self.stream_thread = VideoThread(self.client, self.selected_device_id, target_fps=7, include_audio=True)
        self.stream_thread.frame_ready.connect(self._on_frame)
        self.stream_thread.status_msg.connect(lambda msg: self.video_label.setText(msg))
        self.stream_thread.start()

    def stop_stream(self):
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread.wait()
            self.stream_thread = None

    def _on_frame(self, img):
        if self.stream_thread is not None:
            self.current_frame = self.stream_thread.get_frame_copy()
            self.current_audio = self.stream_thread.get_audio_copy() or b""
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def register_user(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Проверка", "Введите имя пользователя")
            return
        if self.current_frame is None:
            QMessageBox.warning(self, "Проверка", "Ожидаем кадр с камеры")
            return
        ok, enc = cv2.imencode('.jpg', self.current_frame)
        if not ok:
            QMessageBox.warning(self, "Проверка", "Не удалось закодировать кадр")
            return
        self.captured_images.append(enc.tobytes())
        captured = len(self.captured_images)
        self.progress_label.setText(
            f"Снимки: {captured}/{self.required_shots}. Держите лицо под разными углами."
        )
        if captured < self.required_shots:
            self.btn_capture.setText(
                f"📸 Сделать снимок ({captured + 1}/{self.required_shots})"
            )
            return
        if len(self.captured_voices) < self.required_voice_samples:
            QMessageBox.warning(
                self,
                "Проверка",
                f"Запишите голос: {len(self.captured_voices)}/{self.required_voice_samples}",
            )
            return

        try:
            self.client.register_user(name, self.captured_images, self.captured_voices)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка регистрации", str(e))
            self.captured_images.clear()
            self.captured_voices.clear()
            self.progress_label.setText(
                f"Снимки: 0/{self.required_shots}. Сделайте 5 фото под разными ракурсами."
            )
            self.voice_progress_label.setText(
                f"Голос: 0/{self.required_voice_samples}. Запишите короткую фразу для биометрии."
            )
            self.btn_capture.setText(f"📸 Сделать снимок (1/{self.required_shots})")
            self.btn_record_voice.setText(
                f"🎤 Записать голос (1/{self.required_voice_samples})"
            )
            return

        QMessageBox.information(
            self,
            "Готово",
            f"Пользователь зарегистрирован. Снимков: {self.required_shots}, голосовых примеров: {self.required_voice_samples}",
        )
        self.accept()

    def capture_voice_sample(self):
        raw = self.current_audio
        if not raw:
            QMessageBox.warning(self, "Проверка", "Нет аудио из серверного потока камеры/микрофона")
            return
        self.captured_voices.append(raw)
        captured = len(self.captured_voices)
        self.voice_progress_label.setText(
            f"Голос: {captured}/{self.required_voice_samples}. Говорите естественно."
        )
        if captured < self.required_voice_samples:
            self.btn_record_voice.setText(
                f"🎤 Записать голос ({captured + 1}/{self.required_voice_samples})"
            )
        else:
            self.btn_record_voice.setText("🎤 Голос записан")

    def closeEvent(self, event):
        self.stop_stream()
        super().closeEvent(event)


class PersonnelAccessTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.users_cache = []
        self.setup_ui()
        self.refresh_all()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_users_only)
        self.refresh_timer.start(3000)

    def setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(QLabel("<b>Пользователи</b>"))
        self.user_list = QListWidget()
        self.user_list.currentRowChanged.connect(self.load_user_rights)
        left.addWidget(self.user_list)

        actions = QHBoxLayout()
        self.btn_add = QPushButton("➕ Добавить")
        self.btn_add.clicked.connect(self.open_registration_dialog)
        self.btn_del = QPushButton("🗑 Удалить")
        self.btn_del.clicked.connect(self.delete_user)
        self.btn_refresh = QPushButton("🔄 Обновить")
        self.btn_refresh.clicked.connect(self.refresh_all)
        actions.addWidget(self.btn_add)
        actions.addWidget(self.btn_del)
        actions.addWidget(self.btn_refresh)
        left.addLayout(actions)

        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(QLabel("<b>Права доступа</b>"))
        self.rights_tree = QTreeWidget()
        self.rights_tree.setHeaderLabels(["Зона/Комната"])
        right.addWidget(self.rights_tree)
        self.btn_save_rights = QPushButton("💾 Сохранить права")
        self.btn_save_rights.clicked.connect(self.save_rights)
        right.addWidget(self.btn_save_rights)


        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        left_wrap = QWidget(); left_wrap.setLayout(left)
        right_wrap = QWidget(); right_wrap.setLayout(right)
        self.splitter.addWidget(left_wrap)
        self.splitter.addWidget(right_wrap)
        main.addWidget(self.splitter)
        self.apply_layout_ratio()

    def apply_layout_ratio(self):
        total = max(self.width(), 1000)
        self.splitter.setSizes([int(total * (1.0 - INV_PHI)), int(total * INV_PHI)])

    def refresh_users_only(self):
        current_uid = None
        item = self.user_list.currentItem()
        if item:
            current_uid = item.data(Qt.ItemDataRole.UserRole)
        users = list(self.client.list_users())
        if [(u.id, u.name) for u in users] == [(u.id, u.name) for u in self.users_cache]:
            return
        self.users_cache = users
        self.user_list.blockSignals(True)
        self.user_list.clear()
        sel_row = -1
        for idx, u in enumerate(users):
            row = QListWidgetItem(f"#{idx+1} • {u.name} (id={u.id})")
            row.setData(Qt.ItemDataRole.UserRole, u.id)
            self.user_list.addItem(row)
            if current_uid is not None and u.id == current_uid:
                sel_row = idx
        self.user_list.blockSignals(False)
        if sel_row >= 0:
            self.user_list.setCurrentRow(sel_row)
        elif self.user_list.count() > 0:
            self.user_list.setCurrentRow(0)

    def refresh_all(self):
        self.refresh_users_only()
        self.load_user_rights(self.user_list.currentRow())

    def open_registration_dialog(self):
        dlg = RegistrationCameraDialog(self.client, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh_all()

    def delete_user(self):
        item = self.user_list.currentItem()
        if not item:
            return
        uid = item.data(Qt.ItemDataRole.UserRole)
        self.client.remove_user(uid)
        self.refresh_all()

    def load_user_rights(self, _row):
        item = self.user_list.currentItem()
        if not item:
            self.rights_tree.clear()
            return
        uid = item.data(Qt.ItemDataRole.UserRole)
        allowed = set(self.client.get_user_access(uid).allowed_room_ids)
        zones = self.client.list_zones()
        rooms = self.client.list_rooms()
        z_map = {z.id: QTreeWidgetItem([z.name]) for z in zones}
        for r in rooms:
            ri = QTreeWidgetItem([r.name])
            ri.setData(0, Qt.ItemDataRole.UserRole, r.id)
            ri.setCheckState(0, Qt.CheckState.Checked if r.id in allowed else Qt.CheckState.Unchecked)
            if r.zone_id in z_map:
                z_map[r.zone_id].addChild(ri)
        self.rights_tree.clear()
        for z in z_map.values():
            self.rights_tree.addTopLevelItem(z)
            z.setExpanded(True)

    def save_rights(self):
        item = self.user_list.currentItem()
        if not item:
            return
        uid = item.data(Qt.ItemDataRole.UserRole)
        allowed = []
        it = QTreeWidgetItemIterator(self.rights_tree)
        while it.value():
            node = it.value()
            if node.childCount() == 0 and node.checkState(0) == Qt.CheckState.Checked:
                allowed.append(node.data(0, Qt.ItemDataRole.UserRole))
            it += 1
        if len(allowed) > 9:
            QMessageBox.warning(self, "UX-подсказка", "Рекомендуется не более 7±2 комнат на пользователя.")
        self.client.set_access_rules(uid, allowed)
        QMessageBox.information(self, "Готово", "Права сохранены")

class InfrastructureTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.current_dev_id = None
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        tb = QHBoxLayout()
        b1 = QPushButton("+ Зона")
        b1.clicked.connect(self.add_zone)
        tb.addWidget(b1)
        b2 = QPushButton("+ Комната")
        b2.clicked.connect(self.add_room)
        tb.addWidget(b2)
        b3 = QPushButton("+ Устройство")
        b3.clicked.connect(self.add_device)
        tb.addWidget(b3)
        b4 = QPushButton("🔄 Обновить")
        b4.clicked.connect(self.refresh_tree)
        tb.addWidget(b4)
        tb.addStretch()
        left.addLayout(tb)

        et = QHBoxLayout()
        self.btn_del = QPushButton("🗑 Удалить")
        self.btn_del.clicked.connect(self.delete_item)
        self.btn_del.setEnabled(False)
        et.addStretch()
        et.addWidget(self.btn_del)
        left.addLayout(et)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Инфраструктура", "ID", "Статус"])
        self.tree.currentItemChanged.connect(self.on_select)
        left.addWidget(self.tree)
        layout.addLayout(left, 2)

        right = QGroupBox("Детали")
        rl = QVBoxLayout()
        self.lbl_info = QLabel("Выберите элемент...")
        self.lbl_info.setWordWrap(True)
        rl.addWidget(self.lbl_info)
        self.door_controls = QWidget()
        dl = QVBoxLayout(self.door_controls)
        b_o = QPushButton("🔓 Открыть")
        b_o.clicked.connect(lambda: self.control_door("open_once"))
        dl.addWidget(QLabel("<b>Замок:</b>"))
        dl.addWidget(b_o)
        self.door_controls.setVisible(False)
        rl.addWidget(self.door_controls)
        rl.addStretch()
        right.setLayout(rl)
        layout.addWidget(right, 1)
        self.refresh_tree()

    def refresh_tree(self):
        self.tree.clear()
        try:
            zones = self.client.list_zones()
            rooms = self.client.list_rooms()
            devices = self.client.list_devices()
            z_map = {z.id: QTreeWidgetItem([z.name, str(z.id), "Zone"]) for z in zones}
            r_map = {r.id: QTreeWidgetItem([r.name, str(r.id), "Room"]) for r in rooms}
            for z in z_map.values():
                z.setData(0, Qt.ItemDataRole.UserRole, ("zone", 0))
                self.tree.addTopLevelItem(z)
            for r in rooms:
                if r.zone_id in z_map:
                    item = r_map[r.id]
                    item.setData(0, Qt.ItemDataRole.UserRole, ("room", r.id))
                    z_map[r.zone_id].addChild(item)
            for d in devices:
                if d.room_id in r_map:
                    item = QTreeWidgetItem([d.name, str(d.id), d.device_type])
                    item.setData(0, Qt.ItemDataRole.UserRole, ("device", d))
                    r_map[d.room_id].addChild(item)
            self.tree.expandAll()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки инфраструктуры", str(e))

    def _get_detected_hardware(self, device_type: str):
        out = []
        try:
            devs = self.client.scan_hardware()
        except Exception:
            return out
        seen_mic_conn = set()
        for d in devs:
            if d.device_type != device_type or d.id > 0:
                continue
            if device_type == "microphone":
                conn = (d.connection_string or "").strip().lower()
                if not conn:
                    continue
                virtual_tokens = (
                    "null", "loopback", "default", "sysdefault", "front:", "surround",
                    "dmix", "dsnoop", "cards.pcm", "iec958", "hdmi", "usbstream",
                )
                if any(tok in conn for tok in virtual_tokens):
                    continue
                if not (conn.startswith("hw:") or conn.startswith("plughw:")):
                    continue
                if conn in seen_mic_conn:
                    continue
                seen_mic_conn.add(conn)
            out.append(d)
        return out

    def add_zone(self):
        n, ok = QInputDialog.getText(self, "Новая зона", "Название:")
        if ok and n:
            self.client.add_zone(n)
            self.refresh_tree()

    def add_room(self):
        zones = self.client.list_zones()
        d = QDialog(self)
        l = QFormLayout(d)
        ne = QLineEdit()
        zc = QComboBox()
        for z in zones:
            zc.addItem(z.name, z.id)
        l.addRow("Название", ne)
        l.addRow("Зона", zc)
        b = QPushButton("Добавить")
        b.clicked.connect(d.accept)
        l.addRow(b)
        if d.exec() and ne.text():
            self.client.add_room(ne.text(), zc.currentData())
            self.refresh_tree()

    def add_device(self):
        rooms = self.client.list_rooms()
        if not rooms:
            QMessageBox.warning(self, "Проверка", "Сначала создайте комнату")
            return

        d = QDialog(self)
        d.setWindowTitle("Добавить устройство")
        d.resize(600, 260)
        l = QFormLayout(d)
        ne = QLineEdit()
        rc = QComboBox()
        for r in rooms:
            rc.addItem(r.name, r.id)
        tc = QComboBox()
        tc.addItems(["camera", "microphone", "lock"])
        source_box = QComboBox()
        ce = QLineEdit()
        ce.setPlaceholderText("Подключение устройства")
        btn_refresh_sources = QPushButton("Обновить найденные источники")

        def refresh_sources():
            dtype = tc.currentText()
            source_box.clear()
            if dtype == "lock":
                source_box.addItem("Автогенерация пути lock-файла", "")
                ce.setEnabled(False)
                return
            ce.setEnabled(True)
            detected = self._get_detected_hardware(dtype)
            for dev in detected:
                label = f"{dev.name} | {dev.connection_string}"
                source_box.addItem(label, dev.connection_string)
            source_box.addItem("Ввести вручную…", "__manual__")
            if source_box.count() > 0 and source_box.itemData(0) != "__manual__":
                ce.setText(source_box.itemData(0))
            else:
                ce.clear()

        def on_type_changed():
            refresh_sources()
            if tc.currentText() == "lock":
                if not ne.text().strip():
                    ne.setText("lock")
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", ne.text().strip() or "lock")
                ce.setText(f"/workspace/identification/locks/{slug}.lock")

        def on_name_changed():
            if tc.currentText() == "lock":
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", ne.text().strip() or "lock")
                ce.setText(f"/workspace/identification/locks/{slug}.lock")

        def on_source_changed(index):
            if index < 0:
                return
            data = source_box.itemData(index)
            if data == "__manual__":
                ce.setEnabled(True)
                if not ce.text().strip():
                    ce.clear()
                return
            if data:
                ce.setText(str(data))
                ce.setEnabled(False)

        l.addRow("Название", ne)
        l.addRow("Комната", rc)
        l.addRow("Тип", tc)
        l.addRow("Найденный источник", source_box)
        l.addRow("Подключение", ce)
        l.addRow("", btn_refresh_sources)
        b = QPushButton("Добавить")
        b.clicked.connect(d.accept)
        l.addRow(b)
        tc.currentIndexChanged.connect(on_type_changed)
        ne.textChanged.connect(on_name_changed)
        source_box.currentIndexChanged.connect(on_source_changed)
        btn_refresh_sources.clicked.connect(refresh_sources)
        refresh_sources()

        if d.exec() and ne.text().strip():
            dtype = tc.currentText()
            conn = ce.text().strip()
            if dtype != "lock" and not conn:
                QMessageBox.warning(self, "Проверка", "Укажите подключение устройства")
                return
            try:
                self.client.add_device(ne.text().strip(), rc.currentData(), dtype, conn)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка добавления устройства", str(e))
            self.refresh_tree()

    def delete_item(self):
        item = self.tree.currentItem()
        if not item:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "device":
            self.client.remove_device(data[1].id)
            self.refresh_tree()

    def on_select(self, current, _prev):
        self.btn_del.setEnabled(False)
        self.door_controls.setVisible(False)
        if not current:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "device":
            self.btn_del.setEnabled(True)
            self.current_dev_id = data[1].id
            self.lbl_info.setText(f"Устройство: {data[1].name}\nПодключение: {data[1].connection_string}")
            if data[1].device_type == "lock":
                self.door_controls.setVisible(True)

    def control_door(self, cmd):
        if self.current_dev_id is not None:
            self.client.control_door(self.current_dev_id, cmd)

class SettingsTab(QWidget):
    pipeline_state_changed = pyqtSignal(bool)
    settings_changed = pyqtSignal(dict)

    def __init__(self, app, settings, client):
        super().__init__()
        self.app = app
        self.settings = settings
        self.client = client

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.grp_live = QGroupBox("Система")
        self.grp_live.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        live_layout = QGridLayout()
        live_layout.setContentsMargins(10, 10, 10, 10)
        live_layout.setHorizontalSpacing(10)
        live_layout.setVerticalSpacing(8)
        self.lbl_live_title = QLabel("Состояние:")
        self.lbl_live_state = QLabel("⏳ проверка")
        self.lbl_live_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_state.setMinimumWidth(120)
        self.lbl_live_state.setMaximumWidth(150)
        self.lbl_live_state.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.btn_pipeline_start = QPushButton("▶ Запустить систему")
        self.btn_pipeline_stop = QPushButton("■ Остановить систему")
        self.btn_pipeline_start.clicked.connect(self.start_pipeline)
        self.btn_pipeline_stop.clicked.connect(self.stop_pipeline)
        live_layout.addWidget(self.lbl_live_title, 0, 0)
        live_layout.addWidget(self.lbl_live_state, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        live_layout.addWidget(self.btn_pipeline_start, 1, 0)
        live_layout.addWidget(self.btn_pipeline_stop, 1, 1)
        live_layout.setColumnStretch(0, 1)
        live_layout.setColumnStretch(1, 1)
        self.grp_live.setLayout(live_layout)

        self.grp_ui = QGroupBox("Интерфейс")
        self.grp_ui.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        ui_layout = QFormLayout()
        ui_layout.setContentsMargins(10, 10, 10, 10)
        ui_layout.setHorizontalSpacing(12)
        ui_layout.setVerticalSpacing(8)
        self.theme = QComboBox()
        self.theme.addItems(["Светлая", "Тёмная"])
        self.theme.setCurrentText("Тёмная" if settings.get("theme", "light") == "dark" else "Светлая")
        self.font_size = QComboBox()
        self.font_size.addItems(["11", "12", "13", "14", "15", "16", "18"])
        self.font_size.setCurrentText(str(settings.get("font_size", DEFAULT_UI_SETTINGS["font_size"])))
        self.voice_mode = QComboBox()
        self.voice_mode.addItems(["Включено", "Отключено"])
        self.voice_mode.setCurrentText("Включено" if settings.get("voice_modality", True) else "Отключено")
        self.system_log_default = QComboBox()
        self.system_log_default.addItems(["Скрыт", "Показан"])
        self.system_log_default.setCurrentText("Показан" if settings.get("show_system_log", False) else "Скрыт")
        self.btn_apply = QPushButton("💾 Сохранить параметры интерфейса")
        self.btn_apply.clicked.connect(self.apply)
        ui_layout.addRow("Тема", self.theme)
        ui_layout.addRow("Размер шрифта", self.font_size)
        ui_layout.addRow("Голос в мониторинге", self.voice_mode)
        ui_layout.addRow("Системный лог по умолчанию", self.system_log_default)
        ui_layout.addRow(self.btn_apply)
        self.grp_ui.setLayout(ui_layout)

        top.addWidget(self.grp_live, 46)
        top.addWidget(self.grp_ui, 54)
        root.addLayout(top)
        root.addStretch(1)

        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.refresh_pipeline_state)
        self.state_timer.start(1500)
        self.refresh_pipeline_state()

    def set_pipeline_state(self, running: bool):
        self.lbl_live_state.setText("🟢 Работает" if running else "🔴 Остановлен")
        self.lbl_live_state.setObjectName("liveOn" if running else "liveOff")
        self.lbl_live_state.style().unpolish(self.lbl_live_state)
        self.lbl_live_state.style().polish(self.lbl_live_state)
        self.btn_pipeline_start.setEnabled(not running)
        self.btn_pipeline_stop.setEnabled(running)

    def refresh_pipeline_state(self):
        try:
            running = self.client.get_pipeline_running()
        except Exception:
            return
        self.set_pipeline_state(bool(running))
        self.pipeline_state_changed.emit(bool(running))

    def start_pipeline(self):
        try:
            self.client.set_pipeline_running(True)
            self.set_pipeline_state(True)
            self.pipeline_state_changed.emit(True)
            QMessageBox.information(self, "Система", "Система запущена.")
        except Exception as e:
            QMessageBox.critical(self, "Система", str(e))

    def stop_pipeline(self):
        try:
            self.client.set_pipeline_running(False)
            self.set_pipeline_state(False)
            self.pipeline_state_changed.emit(False)
            QMessageBox.information(self, "Система", "Система остановлена.")
        except Exception as e:
            QMessageBox.critical(self, "Система", str(e))

    def apply(self):
        self.settings["theme"] = "dark" if self.theme.currentText() == "Тёмная" else "light"
        self.settings["font_size"] = int(self.font_size.currentText())
        self.settings["voice_modality"] = self.voice_mode.currentText() == "Включено"
        self.settings["show_system_log"] = self.system_log_default.currentText() == "Показан"
        save_ui_settings(self.settings)
        apply_ui_theme(self.app, self.settings)
        self.settings_changed.emit(dict(self.settings))
        QMessageBox.information(self, "Настройки", "Параметры интерфейса сохранены")


class LogTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        layout = QVBoxLayout(self)
        self.btn_refresh = QPushButton("🔄 Обновить журнал")
        self.btn_refresh.clicked.connect(self.load_logs)
        layout.addWidget(self.btn_refresh)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "Время", "Пользователь", "Помещение", "Доступ", "Детали"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(2, 180)
        self.table.setColumnWidth(3, 220)
        self.table.setColumnWidth(5, 620)
        self.table.verticalHeader().setDefaultSectionSize(30)
        layout.addWidget(self.table)
        self.load_logs()

    def load_logs(self):
        self.table.setRowCount(0)
        try:
            logs = self.client.get_logs(limit=100)
            self.table.setRowCount(len(logs))
            for i, log in enumerate(logs):
                ts_fmt = format_iso_to_local(log.timestamp or "")
                self.table.setItem(i, 0, QTableWidgetItem(str(log.id)))
                self.table.setItem(i, 1, QTableWidgetItem(ts_fmt))
                self.table.setItem(i, 2, QTableWidgetItem(log.user_name))
                self.table.setItem(i, 3, QTableWidgetItem(log.room_name))
                self.table.setItem(i, 4, QTableWidgetItem("✅" if log.access_granted else "❌"))
                self.table.setItem(i, 5, QTableWidgetItem(log.details))
        except Exception:
            return

class HelpTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_ref = QPushButton("📘 Открыть справку")
        self.btn_guide = QPushButton("📗 Открыть руководство пользователя")
        self.btn_ref.clicked.connect(lambda: self.load_doc(HELP_REFERENCE_PATH, "Справка"))
        self.btn_guide.clicked.connect(lambda: self.load_doc(USER_GUIDE_PATH, "Руководство пользователя"))
        top.addWidget(self.btn_ref)
        top.addWidget(self.btn_guide)
        top.addStretch()
        layout.addLayout(top)

        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Arial", 12))
        layout.addWidget(self.text)

        self.load_doc(HELP_REFERENCE_PATH, "Справка")

    def load_doc(self, path: Path, title: str):
        if not path.exists():
            self.path_label.setText(f"{title}: файл не найден ({path})")
            self.text.setPlainText(
                f"Файл {path} отсутствует.\n"
                "Создайте документ или восстановите его из репозитория."
            )
            return
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            self.path_label.setText(f"{title}: ошибка чтения ({path})")
            self.text.setPlainText(f"Не удалось прочитать файл:\n{e}")
            return
        self.path_label.setText(f"{title}: {path}")
        self.text.setMarkdown(content)

class MonitoringTab(QWidget):
    def __init__(self, client, ui_settings=None):
        super().__init__()
        self.client = client
        self.ui_settings = ui_settings or {}
        self.active_cameras = {}
        self.device_meta = {}
        self.camera_items = []
        self.selected_device_id = None
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.process_active_mode)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.last_presence_ts = {}
        self.last_pipeline_ts = {}
        self.last_presence_state = {}
        self.pipeline_inflight = {}
        self.pipeline_stage = {}
        self.ident_cooldown_until = {}
        self.await_presence_clear = {}
        self.presence_required_hits = 2
        self.presence_clear_required_misses = 4
        self.presence_clear_log_after_s = 1.8
        self.presence_to_ident_delay_s = 0.35
        self.ident_cooldown_s = 2.5
        self.door_sessions = {}
        self.presence_hits = {}
        self.presence_miss_hits = {}
        self.last_presence_boxes = {}
        self.presence_confirmed_since = {}
        self.log_dedup = {}
        self.vision_presence_cache = {}
        self.vision_presence_interval_s = 0.40
        self.use_voice_modality = bool(self.ui_settings.get("voice_modality", True))
        self.pipeline_running = bool(self.client.get_pipeline_running())
        self.monitoring_enabled = False
        self.selected_room_name = None
        self.selected_zone_name = None
        self.point_log_seen = set()
        self.server_visible_cameras = set()
        self.setup_ui()
        self.apply_user_preferences(self.ui_settings)

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
        left.addStretch()
        layout.addLayout(left, 14)

        center = QVBoxLayout()
        self.video_label = QLabel("Выбери устройство-камеру слева")
        self.video_label.setMinimumSize(800, 600)  # Область предпросмотра 4:3.
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #555;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.video_label)

        layout.addLayout(center, 62)

        right = QVBoxLayout()
        log_actions = QHBoxLayout()
        self.btn_toggle_system_log = QPushButton("📋 Краткий лог системы")
        self.btn_toggle_system_log.clicked.connect(self.toggle_system_log)
        log_actions.addWidget(self.btn_toggle_system_log)
        log_actions.addStretch()
        right.addLayout(log_actions)

        self.pipeline_log = QTextEdit()
        self.pipeline_log.setReadOnly(True)
        self.pipeline_log.setMinimumWidth(420)
        self.pipeline_log.setPlaceholderText("Краткий лог: присутствие, этап, итог решения")
        right.addWidget(self.pipeline_log)

        self.system_log = QTextEdit()
        self.system_log.setReadOnly(True)
        self.system_log.setPlaceholderText("Системный лог по точкам прохода")
        self.system_log.setVisible(False)
        right.addWidget(self.system_log)

        right_wrap = QWidget()
        right_wrap.setLayout(right)
        layout.addWidget(right_wrap, 24)

        self.refresh_tree()
        self._refresh_live_banner()

    def _refresh_live_banner(self):
        if self.pipeline_running:
            self.video_label.setText("Выбери устройство-камеру слева")
        else:
            self.video_label.setText(
                "Система остановлена.\n"
                "Перейдите во вкладку «Настройки» и запустите систему."
            )

    def set_pipeline_running(self, running: bool):
        running = bool(running)
        if self.pipeline_running == running:
            return
        self.pipeline_running = running
        if not running:
            if self.log_timer.isActive():
                self.log_timer.stop()
            self.stop_all_videos()
            self.append_pipeline_log("[SYSTEM] Система остановлена. Мониторинг и идентификация заблокированы.")
        elif self.monitoring_enabled:
            self.append_pipeline_log("[SYSTEM] Система запущена. Мониторинг активирован.")
            if not self.log_timer.isActive():
                self.log_timer.start(250)
            self.refresh_tree()
        self._refresh_live_banner()

    def append_pipeline_log(self, text: str):
        self.pipeline_log.append(text)
        trim_text_edit_lines(self.pipeline_log, PIPELINE_LOG_MAX_LINES)

    @staticmethod
    def _format_ts_local(ts: str) -> str:
        return format_iso_to_local(ts)

    def toggle_system_log(self):
        show = not self.system_log.isVisible()
        self.system_log.setVisible(show)
        self.btn_toggle_system_log.setText(
            "📋 Скрыть системный лог" if show else "📋 Краткий лог системы"
        )

    def apply_user_preferences(self, settings: dict):
        self.use_voice_modality = bool(settings.get("voice_modality", True))
        show_sys = bool(settings.get("show_system_log", False))
        self.system_log.setVisible(show_sys)
        self.btn_toggle_system_log.setText(
            "📋 Скрыть системный лог" if show_sys else "📋 Краткий лог системы"
        )

    def append_system_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.system_log.append(f"[{ts}] {text}")
        trim_text_edit_lines(self.system_log, SYSTEM_LOG_MAX_LINES)

    def append_pipeline_log_dedup(self, key: str, text: str, min_interval_s: float = 1.5):
        now = time.time()
        prev = self.log_dedup.get(key, 0.0)
        if now - prev < max(0.0, min_interval_s):
            return
        self.log_dedup[key] = now
        self.append_pipeline_log(text)

    @staticmethod
    def _normalize_vision_reason(reason: str) -> str:
        txt = (reason or "").strip()
        if "Face quality low: sharpness=" in txt:
            return "низкая резкость кадра (подойдите ближе/улучшите освещение)"
        if txt.startswith("vision_rpc_error:"):
            return "ошибка связи с Vision"
        return txt or "неизвестная причина"

    @staticmethod
    def _normalize_camera_status(msg: str):
        raw = (msg or "").strip()
        if not raw:
            return None, None, 0.0
        # Частая и ожидаемая ситуация при остановке/переподключении: не спамим в лог.
        if "Locally cancelled by application" in raw:
            return None, None, 0.0
        if "Device or resource busy" in raw or "resource busy" in raw:
            return (
                "Камера занята другим процессом. Закройте сторонние приложения, которые используют это устройство.",
                "camera_busy",
                8.0,
            )
        if raw.startswith("video_capture_error:"):
            short = raw.split("\n", 1)[0].replace("video_capture_error:", "").strip()
            return (f"Ошибка видеопотока: {short}", "camera_stream_error", 4.0)
        if raw.startswith("Серверный поток недоступен:"):
            return (raw, "camera_stream_unavailable", 4.0)
        return (raw.splitlines()[0], "camera_status", 3.0)

    def _handle_camera_status(self, dev_id: int, msg: str):
        text, key, ttl = self._normalize_camera_status(msg)
        if not text or not key:
            return
        self.append_pipeline_log_dedup(
            f"{key}:{dev_id}",
            f"[CAMERA:{dev_id}] {text}",
            ttl,
        )

    def detect_person_in_frame(self, frame):
        if frame is None:
            return False, None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = img.shape[:2]
        min_side = max(60, int(min(frame_w, frame_h) * 0.10))
        faces = self.face_detector.detectMultiScale(
            img,
            scaleFactor=1.06,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
        if len(faces) == 0:
            return False, None
        valid = []
        frame_area = float(frame_w * frame_h)
        for (x, y, w, h) in faces:
            area_ratio = (w * h) / frame_area
            aspect = w / max(float(h), 1.0)
            if area_ratio < 0.015:
                continue
            if not (0.65 <= aspect <= 1.45):
                continue
            valid.append((x, y, w, h))
        if not valid:
            return False, None
        best = max(valid, key=lambda b: b[2] * b[3])
        return True, best

    def detect_face_presence_by_vision(self, dev_id, frame, now):
        cached = self.vision_presence_cache.get(dev_id)
        if cached and (now - cached.get("ts", 0.0)) < self.vision_presence_interval_s:
            return cached.get("present", False), cached.get("reason", "")
        if frame is None:
            self.vision_presence_cache[dev_id] = {"ts": now, "present": False, "reason": "no_frame"}
            return False, "no_frame"

        ok, enc = cv2.imencode('.jpg', frame)
        if not ok:
            self.vision_presence_cache[dev_id] = {"ts": now, "present": False, "reason": "encode_failed"}
            return False, "encode_failed"

        try:
            rsp = self.client.vision_stub.ProcessFace(
                biometry_pb2.ImageFrame(content=enc.tobytes()),
                timeout=0.8,
            )
            present = bool(getattr(rsp, "detected", False))
            reason = getattr(rsp, "error_msg", "") or "ok"
        except Exception as e:
            present = False
            reason = f"vision_rpc_error:{e}"
        self.vision_presence_cache[dev_id] = {"ts": now, "present": present, "reason": reason}
        return present, reason

    def _face_points_for_overlay(self, frame, face_box):
        if frame is None or face_box is None:
            return []
        x, y, w, h = face_box
        if w < 20 or h < 20:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return []

        eyes = self.eye_detector.detectMultiScale(
            roi,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(max(10, int(w * 0.08)), max(10, int(h * 0.08))),
        )
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            if cy <= y + int(h * 0.62):
                eye_centers.append((cx, cy, ew * eh))
        eye_centers.sort(key=lambda p: p[2], reverse=True)
        eye_points = []
        if eye_centers:
            eye_centers = sorted(eye_centers[:4], key=lambda p: p[0])
            left = eye_centers[0]
            right = eye_centers[-1]
            if abs(right[0] - left[0]) > max(8, int(w * 0.12)):
                eye_points = [(left[0], left[1]), (right[0], right[1])]

        # Опорные точки (нос и уголки рта) вычисляем из геометрии лица.
        nose = (x + w // 2, y + int(h * 0.58))
        mouth_left = (x + int(w * 0.36), y + int(h * 0.78))
        mouth_right = (x + int(w * 0.64), y + int(h * 0.78))
        brow_left = (x + int(w * 0.34), y + int(h * 0.32))
        brow_right = (x + int(w * 0.66), y + int(h * 0.32))
        return eye_points + [brow_left, brow_right, nose, mouth_left, mouth_right]

    @staticmethod
    def detect_voice_presence(raw):
        if not raw:
            return False, None
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if pcm.size == 0:
            return False, None
        rms = float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))
        return rms > 0.008, raw

    def refresh_tree(self):
        prev_selected = self.selected_device_id
        self.tree.clear()
        self.camera_items = []
        self.device_meta = {}
        self.server_visible_cameras = set()
        try:
            for d in self.client.scan_hardware():
                if d.device_type == "camera":
                    self.server_visible_cameras.add(d.connection_string)
            zones = self.client.list_zones()
            rooms = self.client.list_rooms()
            devices = self.client.list_devices()
            zone_names = {z.id: z.name for z in zones}
            room_names = {r.id: r.name for r in rooms}
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
                    self.camera_items.append((d_item, d))
                    zone_id = next((r.zone_id for r in rooms if r.id == d.room_id), 0)
                    zone_name = zone_names.get(zone_id, "UnknownZone")
                    room_name = room_names.get(d.room_id, "UnknownRoom")
                    self.device_meta[d.id] = f"{zone_name}/{room_name}/{d.name}"

            for z in z_map.values():
                self.tree.addTopLevelItem(z)
                z.setExpanded(True)
            if self.camera_items:
                target_item = self.camera_items[0][0]
                for item, dev in self.camera_items:
                    if prev_selected is not None and dev.id == prev_selected:
                        target_item = item
                        break
                self.tree.setCurrentItem(target_item)
            elif not self.active_cameras:
                self.video_label.setText("Нет подключенных камер в инфраструктуре")
        except Exception as e:
            self.append_pipeline_log(f"[UI] refresh error: {e}")

    def on_select(self, current, _prev=None):
        if not current:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "device":
            self.selected_device_id = None
            self.selected_room_name = None
            self.selected_zone_name = None
            self.video_label.setText("Выбери устройство-камеру слева")
            return

        dev = data[1]
        self.selected_device_id = dev.id
        room_item = current.parent()
        zone_item = room_item.parent() if room_item else None
        self.selected_room_name = room_item.text(0) if room_item else None
        self.selected_zone_name = zone_item.text(0) if zone_item else None
        self.point_log_seen.clear()
        self.pipeline_log.clear()
        if not self.monitoring_enabled or not self.pipeline_running:
            self._refresh_live_banner()
            return
        for cam_id, thread in list(self.active_cameras.items()):
            if cam_id == dev.id:
                continue
            thread.stop()
            thread.wait()
            self.active_cameras.pop(cam_id, None)
        if dev.id not in self.active_cameras:
            self.start_camera_stream(dev)

    def start_camera_stream(self, dev):
        if not self.pipeline_running:
            self._refresh_live_banner()
            return
        conn = str(dev.connection_string or "")
        conn_mapped = f"/dev/video{conn}" if conn.isdigit() else conn
        if conn_mapped not in self.server_visible_cameras and conn not in self.server_visible_cameras:
            self.append_pipeline_log(
                f"[CAMERA:{dev.id}] preview blocked: source '{conn}' not reported by server scan."
            )
            self.video_label.setText(
                f"Источник камеры недоступен на сервере: {conn}\nОбновите инфраструктуру/сканирование."
            )
            return
        t = VideoThread(self.client, dev.id, target_fps=5, include_audio=False)
        t.frame_ready.connect(lambda img, d=dev.id: self._show_frame_for_device(d, img))
        t.status_msg.connect(lambda msg, d=dev.id: self._handle_camera_status(d, msg))
        t.start()
        self.active_cameras[dev.id] = t

    def start_all_cameras(self):
        self.stop_all_videos()
        for _item, dev in self.camera_items:
            self.start_camera_stream(dev)

    def _show_frame_for_device(self, dev_id, img):
        if self.selected_device_id is not None and dev_id != self.selected_device_id:
            return
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def stop_all_videos(self):
        for t in self.active_cameras.values():
            t.stop()
            t.wait()
        self.active_cameras.clear()
        self.door_sessions.clear()
        self.presence_hits.clear()
        self.presence_miss_hits.clear()
        self.last_presence_boxes.clear()
        self.presence_confirmed_since.clear()
        self.log_dedup.clear()
        self.vision_presence_cache.clear()
        self.last_presence_ts.clear()
        self.last_presence_state.clear()
        self.pipeline_inflight.clear()
        self.pipeline_stage.clear()
        self.ident_cooldown_until.clear()
        self.await_presence_clear.clear()

    def restart_videos(self):
        self.set_monitoring_enabled(True)

    def set_monitoring_enabled(self, enabled: bool):
        if enabled:
            self.monitoring_enabled = True
            if self.pipeline_running:
                if not self.log_timer.isActive():
                    self.log_timer.start(250)
                self.refresh_tree()
            else:
                self.stop_all_videos()
                self._refresh_live_banner()
        else:
            self.monitoring_enabled = False
            if self.log_timer.isActive():
                self.log_timer.stop()
            self.stop_all_videos()

    @staticmethod
    def _extract_confidence(details: str, fallback: float = 0.0) -> float:
        if fallback > 0.0:
            return fallback
        m = re.search(r"conf\s*([0-9]+(?:\.[0-9]+)?)", details or "", flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\(([0-9]+(?:\.[0-9]+)?)\)", details or "")
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return fallback
        return fallback

    @staticmethod
    def _pipeline_flags_to_text(flags):
        mapped = []
        for f in flags or []:
            if f.startswith("step_presence"):
                mapped.append("обнаружено лицо")
            elif f.startswith("step_liveness:ok"):
                mapped.append("живость: OK")
            elif f.startswith("step_liveness:fail"):
                mapped.append("живость: FAIL")
            elif f.startswith("step_face_id:ok"):
                mapped.append("идентификация лица: OK")
            elif f.startswith("step_face_id:fail"):
                mapped.append("идентификация лица: FAIL")
            elif f.startswith("step_voice_id:ok"):
                mapped.append("идентификация голоса: OK")
            elif f.startswith("step_voice_id:warn"):
                mapped.append("идентификация голоса: mismatch")
            elif f.startswith("step_voice_id:skip"):
                mapped.append("голос: пропущен")
            elif f.startswith("step_policy:ok"):
                mapped.append("доступ в помещение: OK")
            elif f.startswith("step_policy:deny"):
                mapped.append("доступ в помещение: DENY")
        return " -> ".join(mapped)

    @staticmethod
    def _make_access(user_name: str, granted: bool, message: str, final_confidence: float):
        return SimpleNamespace(
            user_name=user_name or "Unknown",
            granted=bool(granted),
            message=message or "",
            final_confidence=float(final_confidence or 0.0),
        )

    def _ingest_access_logs(self, logs):
        use_point_filter = (
            self.selected_device_id is not None and self.selected_room_name and self.selected_zone_name
        )
        for log in reversed(list(logs)):
            if log.id in self.point_log_seen:
                continue
            if use_point_filter and (
                log.room_name != self.selected_room_name or log.zone_name != self.selected_zone_name
            ):
                continue
            self.point_log_seen.add(log.id)
            ts = self._format_ts_local(log.timestamp)
            self.append_pipeline_log(
                f"[{ts}] {'ДОСТУП' if log.access_granted else 'ОТКАЗ'} | user={log.user_name} | {log.details}"
            )

    def _capture_clip_frames(self, thread, first_frame):
        clip_frames = [first_frame]
        clip_ts = [int(time.time() * 1000)]
        for _ in range(7):
            extra = thread.get_frame_copy()
            if extra is None:
                continue
            ok_extra, enc_extra = cv2.imencode('.jpg', extra)
            if not ok_extra:
                continue
            clip_frames.append(enc_extra.tobytes())
            clip_ts.append(int(time.time() * 1000))
        return clip_frames, clip_ts

    def _access_from_door_response(self, door_resp):
        confidence = self._extract_confidence(
            getattr(door_resp, "reason", ""),
            float(getattr(door_resp, "confidence", 0.0)),
        )
        access = self._make_access(
            user_name=getattr(door_resp, "user_name", "Unknown"),
            granted=getattr(door_resp, "access_granted", False),
            message=getattr(door_resp, "reason", ""),
            final_confidence=confidence,
        )
        raw_flags = list(getattr(door_resp, "flags", []))
        self.append_pipeline_log(f"[PIPELINE] {self._pipeline_flags_to_text(raw_flags)}")
        self.append_pipeline_log(
            f"[DOOR] stage={getattr(door_resp, 'stage', 0)} conf={access.final_confidence:.2f} "
            f"flags={raw_flags} reason={access.message}"
        )
        return access, raw_flags

    def _access_from_v2(self, access_v2):
        confidence = self._extract_confidence(
            getattr(access_v2, "reason", ""),
            float(getattr(access_v2, "confidence", 0.0)),
        )
        return self._make_access(
            user_name=getattr(access_v2, "user_name", "Unknown"),
            granted=getattr(access_v2, "granted", False),
            message=getattr(access_v2, "reason", ""),
            final_confidence=confidence,
        )

    def _log_identification_result(self, dev_id, thread, access):
        msg = f"{access.user_name}: {'OK' if access.granted else 'NO'}\n{access.message}"
        color = (0, 255, 0) if access.granted else (0, 0, 255)
        thread.set_overlay(msg, color)

        details = getattr(access, "message", "")
        conf = self._extract_confidence(details, float(getattr(access, "final_confidence", 0.0)))
        self.append_pipeline_log(
            f"[{dev_id}] {'ДОСТУП' if access.granted else 'ОТКАЗ'} | этап: идентификация | "
            f"уверенность: {conf:.2f} | пользователь: {access.user_name} | причина: {details}"
        )
        point = self.device_meta.get(dev_id, f"device-{dev_id}")
        short_reason = (details or "").strip()
        if len(short_reason) > 120:
            short_reason = short_reason[:117] + "..."
        self.append_system_log(
            f"Точка={point} | {'ДОСТУП' if access.granted else 'ОТКАЗ'} | user={access.user_name} | conf={conf:.2f} | {short_reason}"
        )

    def process_active_mode(self):
        # Клиент НЕ запускает идентификацию: live-пайплайн работает на сервере (door-agent).
        # Во вкладке мониторинга только предпросмотр и чтение журнала событий.
        if not self.monitoring_enabled or not self.pipeline_running:
            return
        try:
            logs = self.client.get_logs(limit=60, offset=0)
        except Exception:
            return
        self._ingest_access_logs(logs)

    def closeEvent(self, event):
        self.stop_all_videos()
        super().closeEvent(event)


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)
    status_msg = pyqtSignal(str)

    def __init__(self, client, dev_id, target_fps=6, include_audio=True):
        super().__init__()
        self.client = client
        self.dev_id = dev_id
        self.target_fps = target_fps
        self.include_audio = include_audio
        self.running = True
        self.overlay = ("", (0, 255, 0), 0)
        self.face_overlay = (None, [], (255, 220, 0), 0)
        self.mutex = QMutex()
        self.current_frame = None
        self.current_audio = b""
        self._stream_call = None

    def run(self):
        try:
            self._stream_call = self.client.stream_device_media(
                device_id=self.dev_id,
                target_fps=self.target_fps,
                include_audio=self.include_audio,
                audio_sample_rate=16000,
                audio_chunk_ms=220,
            )
        except Exception as e:
            self.status_msg.emit(f"Не удалось запустить серверный поток: {e}")
            return

        try:
            for chunk in self._stream_call:
                if not self.running:
                    break
                if getattr(chunk, "message", ""):
                    self.status_msg.emit(chunk.message)

                frame_bytes = bytes(getattr(chunk, "jpeg_frame", b"") or b"")
                if not frame_bytes:
                    self.msleep(80)
                    continue

                frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                if frame is None:
                    self.msleep(40)
                    continue

                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_audio = bytes(getattr(chunk, "audio", b"") or b"")
                self.mutex.unlock()

                if self.overlay[2] > 0:
                    y0, dy = 50, 30
                    for i, line in enumerate(self.overlay[0].split('\n')):
                        cv2.putText(frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.overlay[1], 2)
                    self.overlay = (self.overlay[0], self.overlay[1], self.overlay[2] - 1)
                if self.face_overlay[3] > 0:
                    box, points, color, ttl = self.face_overlay
                    if box is not None:
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    for (px, py) in points:
                        cv2.circle(frame, (px, py), 2, color, -1)
                    self.face_overlay = (box, points, color, ttl - 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(img)
        except grpc.RpcError as e:
            if not self.running and e.code() == grpc.StatusCode.CANCELLED:
                return
            self.status_msg.emit(f"Серверный поток недоступен: {e.code().name} {e.details()}")
        except Exception as e:
            self.status_msg.emit(f"Ошибка чтения потока: {e}")
        finally:
            self._stream_call = None

    def stop(self):
        self.running = False
        if self._stream_call is not None:
            try:
                self._stream_call.cancel()
            except Exception:
                pass

    def set_overlay(self, text, color):
        self.overlay = (text, color, 60)

    def set_face_overlay(self, face_box, face_points, present):
        color = (0, 220, 0) if present else (0, 180, 255)
        self.face_overlay = (face_box, face_points or [], color, 12)

    def get_frame_bytes(self):
        self.mutex.lock()
        f = self.current_frame
        self.mutex.unlock()
        if f is not None:
            _, enc = cv2.imencode('.jpg', f)
            return enc.tobytes()
        return None

    def get_frame_copy(self):
        self.mutex.lock()
        f = self.current_frame.copy() if self.current_frame is not None else None
        self.mutex.unlock()
        return f

    def get_audio_copy(self):
        self.mutex.lock()
        a = bytes(self.current_audio) if self.current_audio else b""
        self.mutex.unlock()
        return a


class AdminApp(QMainWindow):
    def __init__(self, app, client, ui_settings):
        super().__init__()
        self.setWindowTitle("Панель администратора Biometry 2.0")
        self.resize(1280, 840)
        self.client = client
        self.app = app
        self.ui_settings = ui_settings

        self.tabs = QTabWidget()
        self.personnel_tab = PersonnelAccessTab(self.client)
        self.monitor_tab = MonitoringTab(self.client, self.ui_settings)
        self.system_tab = SystemTab(self.client)
        self.settings_tab = SettingsTab(self.app, self.ui_settings, self.client)
        self.settings_tab.pipeline_state_changed.connect(self.on_pipeline_state_changed)
        self.settings_tab.settings_changed.connect(self.on_settings_changed)
        self.tabs.addTab(self.monitor_tab, "📹 Мониторинг")
        self.tabs.addTab(self.system_tab, "⚙️ Система")
        self.tabs.addTab(self.settings_tab, "🎛 Настройки")
        self.tabs.addTab(self.personnel_tab, "👥 Персонал и доступ")
        self.tabs.addTab(InfrastructureTab(self.client), "🏗 Инфраструктура")
        self.tabs.addTab(LogTab(self.client), "📜 Журнал")
        self.tabs.addTab(HelpTab(), "❓ Справка")
        self.tabs.currentChanged.connect(self.on_tab_change)
        self.setCentralWidget(self.tabs)
        self.pipeline_state_timer = QTimer(self)
        self.pipeline_state_timer.timeout.connect(self.refresh_pipeline_state)
        self.pipeline_state_timer.start(1500)
        self.refresh_pipeline_state()
        self.tabs.setCurrentIndex(1)
        self.on_tab_change(1)

    def refresh_pipeline_state(self):
        try:
            running = bool(self.client.get_pipeline_running())
        except Exception:
            return
        self.on_pipeline_state_changed(running)

    def on_pipeline_state_changed(self, running: bool):
        self.monitor_tab.set_pipeline_running(bool(running))
        self.settings_tab.set_pipeline_state(bool(running))

    def on_settings_changed(self, settings: dict):
        self.ui_settings = dict(settings)
        self.monitor_tab.apply_user_preferences(self.ui_settings)

    def on_tab_change(self, index):
        if self.tabs.widget(index) == self.monitor_tab:
            self.monitor_tab.set_monitoring_enabled(True)
        else:
            self.monitor_tab.set_monitoring_enabled(False)

    def closeEvent(self, event):
        self.monitor_tab.set_monitoring_enabled(False)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui_settings = load_ui_settings()
    apply_ui_theme(app, ui_settings)

    gateway_addr = os.getenv("GATEWAY_ADDR", "127.0.0.1:50051")
    if len(sys.argv) > 1 and sys.argv[1].strip():
        gateway_addr = sys.argv[1].strip()

    client = BiometryClient(gateway_addr)
    if not client.wait_until_ready(total_timeout=45.0, probe_timeout=2.0):
        extra_hint = ""
        try:
            # Если docker compose доступен, показываем его вывод прямо в ошибке.
            ps = subprocess.check_output(["docker", "compose", "ps"], text=True, stderr=subprocess.STDOUT)
            extra_hint = (
                "\n\n"
                "Статус docker compose:\n"
                f"{ps}\n"
                "Проверь логи gateway:\n"
                "  docker compose logs --tail=120 gateway-service\n"
            )
        except Exception:
            pass

        QMessageBox.critical(
            None,
            "Шлюз недоступен",
            f"Не удалось подключиться к gRPC Gateway по адресу {gateway_addr}.\n"
            "Проверьте, что gateway-service запущен и порт 50051 проброшен.\n"
            "Docker: ./start_docker.sh\n"
            "По умолчанию клиент использует 127.0.0.1:50051 (или GATEWAY_ADDR)."
            + extra_hint,
        )
        sys.exit(1)

    w = AdminApp(app, client, ui_settings)
    w.show()
    sys.exit(app.exec())
