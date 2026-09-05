"""Replaceable PySide6 presentation layer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .catalog import Catalog
from .domain import SearchResult


class MainWindow(QMainWindow):
    def __init__(self, catalog: Catalog) -> None:
        super().__init__()
        self.catalog = catalog
        self.current_group: str | None = None
        self.current_result: SearchResult | None = None
        self.setWindowTitle("Image Finder — Prototype")
        self.resize(1280, 780)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search subtitle text, OCR text, or filename…")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self.refresh_results)

        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabel("Folders")
        all_item = QTreeWidgetItem(["All images"])
        all_item.setData(0, Qt.ItemDataRole.UserRole, None)
        self.folder_tree.addTopLevelItem(all_item)
        for group in self.catalog.groups():
            item = QTreeWidgetItem([group])
            item.setData(0, Qt.ItemDataRole.UserRole, group)
            self.folder_tree.addTopLevelItem(item)
        self.folder_tree.setCurrentItem(all_item)
        self.folder_tree.currentItemChanged.connect(self.change_group)

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setMovement(QListWidget.Movement.Static)
        self.gallery.setIconSize(QPixmap(220, 124).size())
        self.gallery.setGridSize(QPixmap(250, 176).size())
        self.gallery.setWordWrap(True)
        self.gallery.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gallery.currentItemChanged.connect(self.show_result)

        self.count_label = QLabel()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(self.search_box)
        center_layout.addWidget(self.count_label)
        center_layout.addWidget(self.gallery, 1)

        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(340, 240)
        self.preview.setStyleSheet("background: #17191d; color: #aeb4be; border-radius: 6px;")
        self.preview.setScaledContents(False)
        self.subtitle = QLabel()
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_button = QPushButton("Open original")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_original)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.preview, 1)
        right_layout.addWidget(self.subtitle)
        right_layout.addWidget(self.details)
        right_layout.addWidget(self.open_button)

        splitter = QSplitter()
        splitter.addWidget(self.folder_tree)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([190, 720, 370])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)
        self.statusBar().showMessage("Source images are read-only; displayed data is rebuildable.")
        self.refresh_results()

    def change_group(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        self.current_group = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self.refresh_results()

    def refresh_results(self) -> None:
        results = self.catalog.search(self.search_box.text(), self.current_group)
        self.gallery.clear()
        for result in results:
            item = QListWidgetItem(result.display_text)
            if result.thumbnail_path.is_file():
                item.setIcon(QIcon(str(result.thumbnail_path)))
            item.setData(Qt.ItemDataRole.UserRole, result)
            item.setToolTip(result.relative_path)
            self.gallery.addItem(item)
        self.count_label.setText(f"{len(results)} image{'s' if len(results) != 1 else ''}")

    def show_result(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self.current_result = current.data(Qt.ItemDataRole.UserRole) if current else None
        result = self.current_result
        if result is None:
            self.preview.setText("Select an image")
            self.preview.setPixmap(QPixmap())
            self.subtitle.clear()
            self.details.clear()
            self.open_button.setEnabled(False)
            return

        pixmap = QPixmap(str(result.absolute_path))
        if pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Preview unavailable")
        else:
            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.subtitle.setText(result.display_text)
        confidence = "—" if result.confidence is None else f"{result.confidence:.1%}"
        self.details.setText(
            f"{result.relative_path}\n\n{result.width} × {result.height} · OCR confidence {confidence}\n"
            f"{result.absolute_path}"
        )
        self.open_button.setEnabled(result.absolute_path.is_file())

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self.current_result:
            current = self.gallery.currentItem()
            if current:
                self.show_result(current, None)

    def open_original(self) -> None:
        if self.current_result and self.current_result.absolute_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_result.absolute_path)))


def run(catalog: Catalog, smoke_test: bool = False) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Image Finder")
    window = MainWindow(catalog)
    window.show()
    if smoke_test:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(250, app.quit)
    return app.exec()
