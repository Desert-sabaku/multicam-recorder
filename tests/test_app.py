import sys
from PySide6.QtWidgets import QApplication, QCheckBox
from multicam_recorder.app import RecorderWindow
from multicam_recorder.camera import ROTATION_OPTIONS


def test_recorder_window_rotation_ui():
    app = QApplication.instance() or QApplication(sys.argv)
    window = RecorderWindow()

    # Check rotation options in sidebar
    combo = window.rotation_box
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == list(ROTATION_OPTIONS)
    assert combo.currentText() == "0°"

    # Simulate camera detection
    checkbox0 = QCheckBox("カメラ 0")
    checkbox1 = QCheckBox("カメラ 1")
    window.camera_checks[0] = checkbox0
    window.camera_checks[1] = checkbox1

    # Simulate building preview grid with cameras 0 and 1
    window._build_preview_grid([0, 1])

    # Check that each camera card has a rotation combo box
    assert 0 in window.camera_rotation_boxes
    assert 1 in window.camera_rotation_boxes
    assert window.camera_rotation_boxes[0].currentText() == "0°"
    assert window.camera_rotation_boxes[1].currentText() == "0°"

    # Change global rotation
    window.rotation_box.setCurrentText("90°")
    assert window.camera_rotations[0] == "90°"
    assert window.camera_rotations[1] == "90°"
    assert window.camera_rotation_boxes[0].currentText() == "90°"
    assert window.camera_rotation_boxes[1].currentText() == "90°"

    # Change individual camera rotation
    window.camera_rotation_boxes[0].setCurrentText("180°")
    assert window.camera_rotations[0] == "180°"
    assert window.camera_rotations[1] == "90°"
    assert window.rotation_box.currentText() == "90°"

    window.close()
