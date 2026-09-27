from __future__ import annotations

import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from .camera import CameraSettings, CameraWorker, discover_camera_indices
from .session import CameraResult, make_session_directory, write_metadata


STYLESHEET = """
QMainWindow, QWidget#root { background: #111827; color: #f9fafb; }
QFrame#sidebar, QLabel#footer { background: #1f2937; }
QLabel { color: #f9fafb; font-family: "Segoe UI"; }
QLabel#muted { color: #9ca3af; }
QLineEdit, QComboBox { background: #111827; color: #f9fafb; border: 1px solid #4b5563; border-radius: 5px; padding: 7px; }
QComboBox QAbstractItemView { background: #1f2937; color: #f9fafb; }
QCheckBox { color: #f9fafb; padding: 5px 2px; }
QPushButton { background: #374151; color: white; border: 0; border-radius: 5px; padding: 9px; font-family: "Segoe UI"; }
QPushButton:hover { background: #4b5563; }
QPushButton:disabled { background: #374151; color: #9ca3af; }
QPushButton#primary { background: #2563eb; font-weight: 600; }
QPushButton#record { background: #dc2626; font-weight: 700; padding: 10px 18px; }
QFrame#previewCard { background: #111827; border: 1px solid #374151; border-radius: 5px; }
QLabel#preview { background: #030712; color: #9ca3af; border: 0; }
"""


class ScannerSignals(QObject):
    finished = Signal(list)


class RecorderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MultiCam Recorder")
        self.resize(1180, 760)
        self.setMinimumSize(820, 560)
        self.camera_checks: dict[int, QCheckBox] = {}
        self.workers: dict[int, CameraWorker] = {}
        self.preview_labels: dict[int, QLabel] = {}
        self.record_state = "idle"
        self.record_start_event: threading.Event | None = None
        self.armed_events: list[threading.Event] = []
        self.arm_deadline = 0.0
        self.session_directory: Path | None = None
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.closing_requested = False
        self.allow_close = False
        self.scanner_signals = ScannerSignals()
        self.scanner_signals.finished.connect(self._show_discovered_cameras)
        self._build_ui()
        self._scan_cameras()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_previews)
        self.preview_timer.start(33)

    def _build_ui(self) -> None:
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 12)
        outer.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("MultiCam Recorder")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        header.addWidget(title)
        header.addStretch()
        self.record_button = QPushButton("● 録画開始", objectName="record")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._toggle_recording)
        header.addWidget(self.record_button)
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet("color: #fca5a5; font: 700 18px Consolas;")
        header.addWidget(self.elapsed_label)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(14)
        outer.addLayout(content, stretch=1)
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(280)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 14, 14, 14)
        content.addWidget(sidebar)
        camera_heading = QLabel("カメラ")
        camera_heading.setStyleSheet("font-size: 15px; font-weight: 700;")
        side.addWidget(camera_heading)
        self.camera_list_layout = QVBoxLayout()
        self.camera_list_layout.setSpacing(1)
        side.addLayout(self.camera_list_layout)
        self.scan_button = QPushButton("カメラを再検出")
        self.scan_button.clicked.connect(self._scan_cameras)
        side.addWidget(self.scan_button)
        side.addSpacing(12)
        side.addWidget(self._section_label("画質"))
        self.resolution_box = QComboBox()
        self.resolution_box.addItems(["640 x 480", "1280 x 720", "1920 x 1080"])
        self.resolution_box.setCurrentText("1280 x 720")
        side.addWidget(self.resolution_box)
        side.addWidget(self._section_label("FPS"))
        self.fps_box = QComboBox()
        self.fps_box.addItems(["15", "24", "30", "60"])
        self.fps_box.setCurrentText("30")
        side.addWidget(self.fps_box)
        side.addSpacing(12)
        side.addWidget(self._section_label("保存先"))
        self.output_edit = QLineEdit(str(Path.cwd() / "recordings"))
        side.addWidget(self.output_edit)
        choose_button = QPushButton("フォルダーを選択")
        choose_button.clicked.connect(self._choose_output)
        side.addWidget(choose_button)
        side.addStretch()
        self.preview_button = QPushButton("プレビュー開始", objectName="primary")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._start_preview)
        side.addWidget(self.preview_button)

        self.preview_container = QWidget()
        self.preview_grid = QGridLayout(self.preview_container)
        self.preview_grid.setContentsMargins(0, 0, 0, 0)
        self.preview_grid.setSpacing(10)
        self.empty_label = QLabel("カメラを選択して「プレビュー開始」を押してください", objectName="muted")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_grid.addWidget(self.empty_label, 0, 0)
        content.addWidget(self.preview_container, stretch=1)
        self.status_label = QLabel("カメラを検出しています…", objectName="footer")
        self.status_label.setContentsMargins(12, 8, 12, 8)
        outer.addWidget(self.status_label)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size: 14px; font-weight: 600; margin-top: 4px;")
        return label

    def _scan_cameras(self) -> None:
        if self.workers:
            QMessageBox.information(self, "確認", "再検出するにはアプリを再起動してください。")
            return
        self.scan_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.status_label.setText("カメラを検出しています…")
        threading.Thread(target=self._scan_in_background, daemon=True).start()

    def _scan_in_background(self) -> None:
        self.scanner_signals.finished.emit(discover_camera_indices())

    def _clear_layout(self, layout: QGridLayout | QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            if item.layout():
                self._clear_layout(item.layout())

    def _show_discovered_cameras(self, indices: list[int]) -> None:
        self._clear_layout(self.camera_list_layout)
        self.camera_checks.clear()
        for index in indices:
            checkbox = QCheckBox(f"カメラ {index}")
            checkbox.setChecked(True)
            self.camera_checks[index] = checkbox
            self.camera_list_layout.addWidget(checkbox)
        if not indices:
            label = QLabel("カメラが見つかりません")
            label.setStyleSheet("color: #fca5a5;")
            self.camera_list_layout.addWidget(label)
            self.status_label.setText("カメラが見つかりません。接続とWindowsのカメラ権限を確認してください。")
        else:
            self.status_label.setText(f"{len(indices)} 台のカメラを検出しました")
        self.scan_button.setEnabled(True)
        self.preview_button.setEnabled(bool(indices))

    def _start_preview(self) -> None:
        selected = [index for index, checkbox in self.camera_checks.items() if checkbox.isChecked()]
        if not selected:
            QMessageBox.warning(self, "カメラ未選択", "1台以上のカメラを選択してください。")
            return
        self._stop_workers()
        width, height = (int(part.strip()) for part in self.resolution_box.currentText().split("x"))
        fps = float(self.fps_box.currentText())
        for index in selected:
            worker = CameraWorker(CameraSettings(index=index, width=width, height=height, fps=fps))
            self.workers[index] = worker
            worker.start()
        self._build_preview_grid(selected)
        self.record_button.setEnabled(True)
        self.preview_button.setText("プレビュー再起動")
        self.status_label.setText(f"{len(selected)} 台のプレビューを開始しています…")

    def _build_preview_grid(self, indices: list[int]) -> None:
        self._clear_layout(self.preview_grid)
        self.preview_labels.clear()
        columns = max(1, math.ceil(math.sqrt(len(indices))))
        for position, index in enumerate(indices):
            card = QFrame(objectName="previewCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(1, 1, 1, 1)
            layout.setSpacing(0)
            preview = QLabel("接続中…", objectName="preview")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(240, 135)
            preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            layout.addWidget(preview, stretch=1)
            name = QLabel(f"カメラ {index}")
            name.setContentsMargins(8, 6, 8, 6)
            layout.addWidget(name)
            self.preview_grid.addWidget(card, position // columns, position % columns)
            self.preview_labels[index] = preview
        for column in range(columns):
            self.preview_grid.setColumnStretch(column, 1)

    def _refresh_previews(self) -> None:
        connected_count = 0
        for index, worker in list(self.workers.items()):
            label = self.preview_labels.get(index)
            if label is None:
                continue
            frame = worker.latest_frame()
            snapshot = worker.snapshot()
            if snapshot.connected:
                connected_count += 1
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
                label.setPixmap(QPixmap.fromImage(image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                label.setText("")
            elif snapshot.error:
                label.setText(snapshot.error)
        if self.workers and self.record_state == "idle":
            self.status_label.setText(f"{connected_count}/{len(self.workers)} 台をプレビュー中")
        if self.record_state == "recording" and self.started_at:
            seconds = max(0, int((datetime.now().astimezone() - self.started_at).total_seconds()))
            self.elapsed_label.setText(f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")

    def _toggle_recording(self) -> None:
        if self.record_state in {"arming", "recording", "stopping"}:
            self._stop_recording()
        else:
            self._begin_recording()

    def _begin_recording(self) -> None:
        ready = {index: worker for index, worker in self.workers.items() if worker.snapshot().connected}
        if len(ready) != len(self.workers) or not ready:
            QMessageBox.warning(self, "準備未完了", "すべてのカメラ映像が表示されてから録画を開始してください。")
            return
        try:
            self.session_directory = make_session_directory(Path(self.output_edit.text()).expanduser().resolve())
        except OSError as exc:
            QMessageBox.critical(self, "保存先エラー", f"保存フォルダーを作成できません。\n{exc}")
            return
        self.record_start_event = threading.Event()
        self.armed_events = []
        try:
            for index, worker in ready.items():
                path = self.session_directory / f"camera_{index:02d}.mp4"
                self.armed_events.append(worker.arm_recording(path, self.record_start_event))
        except RuntimeError as exc:
            for worker in ready.values():
                worker.request_stop_recording()
            QMessageBox.critical(self, "録画エラー", str(exc))
            return
        self.record_state = "arming"
        self.record_button.setText("準備中…")
        self.record_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.status_label.setText("録画ファイルを準備しています…")
        self.arm_deadline = time.monotonic() + 5.0
        QTimer.singleShot(20, self._check_armed)

    def _check_armed(self) -> None:
        failed = [snap.error for worker in self.workers.values() if (snap := worker.snapshot()).error]
        if failed:
            self._cancel_arming("\n".join(failed))
        elif all(event.is_set() for event in self.armed_events):
            self.started_at = datetime.now().astimezone()
            assert self.record_start_event is not None
            self.record_start_event.set()
            self.record_state = "recording"
            self.record_button.setText("■ 録画停止")
            self.record_button.setEnabled(True)
            self.status_label.setText(f"録画中: {self.session_directory}")
        elif time.monotonic() >= self.arm_deadline:
            self._cancel_arming("録画の準備がタイムアウトしました。")
        else:
            QTimer.singleShot(20, self._check_armed)

    def _cancel_arming(self, reason: str) -> None:
        for worker in self.workers.values():
            worker.request_stop_recording()
        self.record_state = "idle"
        self.record_button.setText("● 録画開始")
        self.record_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        QMessageBox.critical(self, "録画エラー", reason)

    def _stop_recording(self) -> None:
        if self.record_state in {"idle", "stopping"}:
            return
        for worker in self.workers.values():
            worker.request_stop_recording()
        self.ended_at = datetime.now().astimezone()
        self.record_state = "stopping"
        self.record_button.setText("保存中…")
        self.record_button.setEnabled(False)
        self.status_label.setText("録画ファイルを閉じています…")
        QTimer.singleShot(30, self._finish_recording)

    def _finish_recording(self) -> None:
        if any(worker.snapshot().recording_active for worker in self.workers.values()):
            QTimer.singleShot(30, self._finish_recording)
            return
        results = []
        for index, worker in self.workers.items():
            snap = worker.snapshot()
            results.append(CameraResult(index, f"camera_{index:02d}.mp4", snap.width, snap.height, snap.fps, snap.frames_written, snap.error))
        if self.session_directory and self.started_at and self.ended_at:
            write_metadata(self.session_directory, self.started_at, self.ended_at, results)
        saved_to = self.session_directory
        self.record_state = "idle"
        self.started_at = self.ended_at = None
        self.elapsed_label.setText("00:00:00")
        self.record_button.setText("● 録画開始")
        self.record_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.status_label.setText(f"録画を保存しました: {saved_to}")
        if self.closing_requested:
            self.allow_close = True
            self.close()

    def _choose_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "録画の保存先", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _stop_workers(self) -> None:
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.allow_close or self.record_state == "idle":
            self.preview_timer.stop()
            self._stop_workers()
            event.accept()
        elif self.record_state == "stopping":
            self.closing_requested = True
            self.status_label.setText("保存完了後に終了します…")
            event.ignore()
        else:
            answer = QMessageBox.question(self, "録画中", "録画を停止して終了しますか？")
            if answer == QMessageBox.StandardButton.Yes:
                self.closing_requested = True
                self._stop_recording()
            event.ignore()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = RecorderWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
