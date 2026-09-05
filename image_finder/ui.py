"""Replaceable PySide6 presentation layer."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis_queue import AnalysisQueue
from .analysis_specs import MOBILE_SUBTITLE_SPEC
from .catalog import Catalog
from .domain import SearchResult


class GalleryModel(QAbstractListModel):
    """Virtual gallery model: icons are decoded only when a view requests them."""

    def __init__(self, show_filename: bool = False) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self.icon_cache: dict[Path, QIcon] = {}
        self.show_filename = show_filename

    def rowCount(self, _parent: QModelIndex = QModelIndex()) -> int:
        return len(self.results)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[no-untyped-def]
        if not index.isValid() or not 0 <= index.row() < len(self.results):
            return None
        result = self.results[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            if self.show_filename:
                return Path(result.relative_path).name
            marker = "✓ " if result.analysis_run_id == MOBILE_SUBTITLE_SPEC.run_id else ""
            return marker + result.display_text
        if role == Qt.ItemDataRole.DecorationRole and result.thumbnail_path.is_file():
            icon = self.icon_cache.get(result.thumbnail_path)
            if icon is None:
                icon = QIcon(str(result.thumbnail_path))
                self.icon_cache[result.thumbnail_path] = icon
            return icon
        if role == Qt.ItemDataRole.ToolTipRole:
            return result.relative_path
        if role == Qt.ItemDataRole.UserRole:
            return result
        return None

    def replace(self, results: list[SearchResult]) -> None:
        self.beginResetModel()
        self.results = results
        self.endResetModel()


class ScanWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, database_path: Path, project_root: Path, library_root: Path) -> None:
        super().__init__()
        self.database_path = database_path
        self.project_root = project_root
        self.library_root = library_root

    @Slot()
    def run(self) -> None:
        worker_catalog = Catalog(self.database_path, self.project_root)
        try:
            summary = worker_catalog.scan_library(
                self.library_root,
                lambda update: self.progress.emit(update.discovered, update.current_path),
            )
            self.finished.emit(summary)
        except Exception as error:  # surfaced to the UI rather than lost in the thread
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            worker_catalog.close()


class OcrWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        database_path: Path,
        project_root: Path,
        batch_size: int,
        ordered_image_ids: list[str],
    ) -> None:
        super().__init__()
        self.database_path = database_path
        self.project_root = project_root
        self.batch_size = batch_size
        self.ordered_image_ids = ordered_image_ids
        self.pause_requested = threading.Event()

    def request_pause(self) -> None:
        self.pause_requested.set()

    @Slot()
    def run(self) -> None:
        worker_catalog = Catalog(self.database_path, self.project_root)
        try:
            queue = AnalysisQueue(worker_catalog.connection)
            run_id = queue.ensure_run(MOBILE_SUBTITLE_SPEC)
            queue.recover_interrupted(run_id)
            batch_ids = queue.prepare_ordered_batch(
                run_id, self.ordered_image_ids, self.batch_size
            )
            starting_counts = queue.counts(run_id)
            target = len(batch_ids)
            if target == 0:
                self.finished.emit(
                    {"processed": 0, "paused": False, "counts": starting_counts}
                )
                return

            from .ocr_engine import PaddleSubtitleOcr

            engine = PaddleSubtitleOcr(self.project_root / "models")
            processed = 0
            processed_ids: list[str] = []
            while processed < target and not self.pause_requested.is_set():
                job = queue.claim_next(run_id, batch_ids)
                if job is None:
                    break
                try:
                    output = engine.analyze(job.source_path)
                    queue.complete(
                        job,
                        all_text=output.all_text,
                        subtitle_text=output.subtitle_text,
                        confidence=output.confidence,
                        payload=output.payload,
                    )
                    processed += 1
                    processed_ids.append(job.image_id)
                    self.progress.emit(processed, job.source_path.name)
                except Exception as error:
                    queue.fail(job, f"{type(error).__name__}: {error}")
                    processed += 1
                    processed_ids.append(job.image_id)
                    self.progress.emit(processed, f"Failed: {job.source_path.name}")
            self.finished.emit(
                {
                    "processed": processed,
                    "paused": self.pause_requested.is_set(),
                    "counts": queue.counts(run_id),
                    "processed_ids": processed_ids,
                }
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            worker_catalog.close()


class MainWindow(QMainWindow):
    def __init__(self, catalog: Catalog) -> None:
        super().__init__()
        self.catalog = catalog
        self.current_group: str | None = None
        self.current_result: SearchResult | None = None
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.ocr_thread: QThread | None = None
        self.ocr_worker: OcrWorker | None = None
        self.ocr_batch_size = 10
        self.setWindowTitle("Image Finder — Prototype")
        self.resize(1280, 780)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search subtitle text, OCR text, or filename…")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self.refresh_results)

        self.scan_button = QPushButton("Add / scan folder…")
        self.scan_button.clicked.connect(self.choose_library)
        search_row = QWidget()
        search_layout = QHBoxLayout(search_row)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self.search_box, 1)
        search_layout.addWidget(self.scan_button)

        self.ocr_button = QPushButton(f"OCR next {self.ocr_batch_size} in view")
        self.ocr_button.clicked.connect(self.start_ocr_batch)
        self.pause_ocr_button = QPushButton("Pause OCR")
        self.pause_ocr_button.setEnabled(False)
        self.pause_ocr_button.clicked.connect(self.pause_ocr)
        self.retry_ocr_button = QPushButton("Retry failed")
        self.retry_ocr_button.clicked.connect(self.retry_failed_ocr)
        self.recent_ocr_button = QPushButton("Last OCR batch")
        self.recent_ocr_button.setCheckable(True)
        self.recent_ocr_button.toggled.connect(lambda _checked: self.refresh_results())
        self.ocr_status = QLabel()
        ocr_row = QWidget()
        ocr_layout = QHBoxLayout(ocr_row)
        ocr_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(self.ocr_button)
        ocr_layout.addWidget(self.pause_ocr_button)
        ocr_layout.addWidget(self.retry_ocr_button)
        ocr_layout.addWidget(self.recent_ocr_button)
        ocr_layout.addWidget(self.ocr_status, 1)

        self.ocr_preview_label = QLabel("Next to OCR")
        self.ocr_preview_model = GalleryModel(show_filename=True)
        self.ocr_preview = QListView()
        self.ocr_preview.setModel(self.ocr_preview_model)
        self.ocr_preview.setViewMode(QListView.ViewMode.IconMode)
        self.ocr_preview.setFlow(QListView.Flow.LeftToRight)
        self.ocr_preview.setWrapping(False)
        self.ocr_preview.setMovement(QListView.Movement.Static)
        self.ocr_preview.setResizeMode(QListView.ResizeMode.Adjust)
        self.ocr_preview.setIconSize(QPixmap(112, 63).size())
        self.ocr_preview.setGridSize(QPixmap(132, 94).size())
        self.ocr_preview.setFixedHeight(112)
        self.ocr_preview.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.ocr_preview.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ocr_preview.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )

        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderLabel("Folders")
        all_item = QTreeWidgetItem(["All images"])
        all_item.setData(0, Qt.ItemDataRole.UserRole, None)
        self.folder_tree.addTopLevelItem(all_item)
        self.populate_groups()
        self.folder_tree.setCurrentItem(all_item)
        self.folder_tree.currentItemChanged.connect(self.change_group)

        self.gallery_model = GalleryModel()
        self.gallery = QListView()
        self.gallery.setModel(self.gallery_model)
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setIconSize(QPixmap(220, 124).size())
        self.gallery.setGridSize(QPixmap(250, 176).size())
        self.gallery.setWordWrap(True)
        self.gallery.setUniformItemSizes(True)
        self.gallery.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gallery.selectionModel().currentChanged.connect(self.show_result)

        self.count_label = QLabel()
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 0)
        self.scan_progress.hide()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(search_row)
        center_layout.addWidget(ocr_row)
        center_layout.addWidget(self.ocr_preview_label)
        center_layout.addWidget(self.ocr_preview)
        center_layout.addWidget(self.scan_progress)
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
        self.analysis_selector = QComboBox()
        self.analysis_selector.currentIndexChanged.connect(self.show_analysis_version)
        self.analysis_details = QLabel()
        self.analysis_details.setWordWrap(True)
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
        right_layout.addWidget(self.analysis_selector)
        right_layout.addWidget(self.analysis_details)
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
        self.refresh_ocr_status()

    def populate_groups(self) -> None:
        while self.folder_tree.topLevelItemCount() > 1:
            self.folder_tree.takeTopLevelItem(1)
        for group in self.catalog.groups():
            item = QTreeWidgetItem([group])
            item.setData(0, Qt.ItemDataRole.UserRole, group)
            self.folder_tree.addTopLevelItem(item)

    def choose_library(self) -> None:
        roots = self.catalog.library_roots()
        starting_directory = str(roots[-1]) if roots else str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self, "Select a read-only image library", starting_directory
        )
        if selected:
            self.start_scan(Path(selected))

    def start_scan(self, library_root: Path) -> None:
        if self.scan_thread and self.scan_thread.isRunning():
            return
        self.scan_button.setEnabled(False)
        self.ocr_button.setEnabled(False)
        self.retry_ocr_button.setEnabled(False)
        self.scan_progress.setRange(0, 0)
        self.scan_progress.show()
        self.statusBar().showMessage(f"Scanning {library_root}…")

        self.scan_thread = QThread(self)
        self.scan_worker = ScanWorker(
            self.catalog.database_path, self.catalog.project_root, library_root
        )
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.update_scan_progress)
        self.scan_worker.finished.connect(self.finish_scan)
        self.scan_worker.failed.connect(self.fail_scan)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.failed.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.start()

    def update_scan_progress(self, count: int, relative_path: str) -> None:
        self.statusBar().showMessage(f"Discovered {count:,} images · {relative_path}")

    def finish_scan(self, summary: object) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.ocr_button.setEnabled(True)
        self.retry_ocr_button.setEnabled(True)
        self.populate_groups()
        self.refresh_results()
        self.statusBar().showMessage(
            f"Scan complete · {summary.discovered:,} images · "
            f"{summary.hashed:,} hashed · {summary.unchanged:,} unchanged · "
            f"{summary.errors:,} errors"
        )
        self.scan_worker = None
        self.scan_thread = None

    def fail_scan(self, message: str) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.ocr_button.setEnabled(True)
        self.retry_ocr_button.setEnabled(True)
        self.statusBar().showMessage("Scan failed")
        QMessageBox.critical(self, "Scan failed", message)
        self.scan_worker = None
        self.scan_thread = None

    def refresh_ocr_status(self) -> None:
        counts = AnalysisQueue(self.catalog.connection).counts(MOBILE_SUBTITLE_SPEC.run_id)
        self.ocr_status.setText(
            f"OCR: {counts['succeeded']:,} complete · {counts['pending']:,} pending · "
            f"{counts['failed']:,} failed"
        )
        self.retry_ocr_button.setEnabled(
            counts["failed"] > 0 and not (self.ocr_thread and self.ocr_thread.isRunning())
        )

    def refresh_ocr_preview(self) -> None:
        if self.ocr_thread and self.ocr_thread.isRunning():
            return
        ordered_results: dict[str, SearchResult] = {}
        for result in self.gallery_model.results:
            ordered_results.setdefault(result.image_id, result)
        queue = AnalysisQueue(self.catalog.connection)
        preview_ids, total = queue.preview_ordered_batch(
            MOBILE_SUBTITLE_SPEC.run_id,
            list(ordered_results),
            self.ocr_batch_size,
        )
        self.ocr_preview_model.replace(
            [ordered_results[image_id] for image_id in preview_ids]
        )
        batches = (total + self.ocr_batch_size - 1) // self.ocr_batch_size
        if total:
            self.ocr_preview_label.setText(
                f"Next to OCR · showing {len(preview_ids)} of {total:,} eligible "
                f"in this view · about {batches:,} batch{'es' if batches != 1 else ''}"
            )
            self.ocr_button.setEnabled(
                not (self.scan_thread and self.scan_thread.isRunning())
            )
        else:
            self.ocr_preview_label.setText("Next to OCR · no eligible images in this view")
            self.ocr_button.setEnabled(False)

    def start_ocr_batch(self) -> None:
        if self.ocr_thread and self.ocr_thread.isRunning():
            return
        self.scan_button.setEnabled(False)
        self.ocr_button.setEnabled(False)
        self.retry_ocr_button.setEnabled(False)
        self.pause_ocr_button.setEnabled(True)
        self.scan_progress.setRange(0, self.ocr_batch_size)
        self.scan_progress.setValue(0)
        self.scan_progress.show()
        self.statusBar().showMessage("Preparing local OCR models…")
        self.ocr_preview_label.setText(
            f"Current OCR batch · {self.ocr_preview_model.rowCount()} image(s)"
        )

        self.ocr_thread = QThread(self)
        ordered_image_ids = [result.image_id for result in self.gallery_model.results]
        self.ocr_worker = OcrWorker(
            self.catalog.database_path,
            self.catalog.project_root,
            self.ocr_batch_size,
            ordered_image_ids,
        )
        self.ocr_worker.moveToThread(self.ocr_thread)
        self.ocr_thread.started.connect(self.ocr_worker.run)
        self.ocr_worker.progress.connect(self.update_ocr_progress)
        self.ocr_worker.finished.connect(self.finish_ocr)
        self.ocr_worker.failed.connect(self.fail_ocr)
        self.ocr_worker.finished.connect(self.ocr_thread.quit)
        self.ocr_worker.failed.connect(self.ocr_thread.quit)
        self.ocr_thread.finished.connect(self.ocr_worker.deleteLater)
        self.ocr_thread.finished.connect(self.ocr_thread.deleteLater)
        self.ocr_thread.start()

    def pause_ocr(self) -> None:
        if self.ocr_worker:
            self.ocr_worker.request_pause()
            self.pause_ocr_button.setEnabled(False)
            self.statusBar().showMessage("Pausing after the current image…")

    def update_ocr_progress(self, processed: int, filename: str) -> None:
        self.scan_progress.setValue(processed)
        self.statusBar().showMessage(
            f"OCR batch {processed}/{self.ocr_batch_size} · {filename}"
        )
        self.refresh_ocr_status()

    def finish_ocr(self, summary: object) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.ocr_button.setEnabled(True)
        self.pause_ocr_button.setEnabled(False)
        self.refresh_ocr_status()
        self.recent_ocr_button.setChecked(True)
        self.refresh_results()
        if summary["processed"] == 0:
            self.statusBar().showMessage("No unprocessed images in the current view")
        else:
            state = "paused" if summary["paused"] else "complete"
            self.statusBar().showMessage(
                f"OCR batch {state} · {summary['processed']} image(s) processed"
            )
        self.ocr_worker = None
        self.ocr_thread = None

    def fail_ocr(self, message: str) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.ocr_button.setEnabled(True)
        self.pause_ocr_button.setEnabled(False)
        self.refresh_ocr_status()
        self.refresh_ocr_preview()
        self.statusBar().showMessage("OCR worker failed")
        QMessageBox.critical(self, "OCR worker failed", message)
        self.ocr_worker = None
        self.ocr_thread = None

    def retry_failed_ocr(self) -> None:
        queue = AnalysisQueue(self.catalog.connection)
        retried = queue.retry_failed(MOBILE_SUBTITLE_SPEC.run_id)
        if retried:
            self.start_ocr_batch()
        else:
            self.refresh_ocr_status()

    def change_group(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        self.current_group = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self.refresh_results()

    def refresh_results(self) -> None:
        results = self.catalog.search(self.search_box.text(), self.current_group)
        if self.recent_ocr_button.isChecked():
            recent_ids = self.catalog.latest_analysis_batch_image_ids(
                MOBILE_SUBTITLE_SPEC.run_id
            )
            results = [result for result in results if result.image_id in recent_ids]
        self.gallery_model.replace(results)
        self.count_label.setText(f"{len(results)} image{'s' if len(results) != 1 else ''}")
        self.refresh_ocr_preview()

    def show_result(self, current: QModelIndex, _previous: QModelIndex) -> None:
        self.current_result = (
            self.gallery_model.data(current, Qt.ItemDataRole.UserRole)
            if current.isValid()
            else None
        )
        result = self.current_result
        if result is None:
            self.preview.setText("Select an image")
            self.preview.setPixmap(QPixmap())
            self.subtitle.clear()
            self.analysis_selector.clear()
            self.analysis_details.clear()
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
        self.details.setText(
            f"{result.relative_path}\n\n{result.width} × {result.height}\n\n"
            f"{result.absolute_path}"
        )
        history = self.catalog.analysis_history(result.image_id)
        self.analysis_selector.blockSignals(True)
        self.analysis_selector.clear()
        for record in history:
            status = (
                "Current OCR"
                if record.run_id == MOBILE_SUBTITLE_SPEC.run_id
                else "Earlier analysis"
            )
            self.analysis_selector.addItem(
                f"{status} · {record.engine_name} · {record.created_at[:10]}", record
            )
        if not history:
            self.analysis_selector.addItem("No analysis yet", None)
        self.analysis_selector.blockSignals(False)
        self.analysis_selector.setCurrentIndex(0)
        self.show_analysis_version(0)
        self.open_button.setEnabled(result.absolute_path.is_file())

    def show_analysis_version(self, index: int) -> None:
        record = self.analysis_selector.itemData(index) if index >= 0 else None
        if record is None:
            self.subtitle.setText("(No recognized text)")
            self.analysis_details.setText("Current OCR version: not processed")
            return
        self.subtitle.setText(record.display_text)
        confidence = "—" if record.confidence is None else f"{record.confidence:.1%}"
        current = (
            "Current OCR version"
            if record.run_id == MOBILE_SUBTITLE_SPEC.run_id
            else "Earlier analysis version"
        )
        self.analysis_details.setText(
            f"{current} · confidence {confidence}\n"
            f"{record.engine_name} {record.engine_version} · {record.model_name}\n"
            f"Pipeline: {record.pipeline_version}\nCompleted: {record.created_at}"
        )

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self.current_result:
            current = self.gallery.currentIndex()
            if current.isValid():
                self.show_result(current, QModelIndex())

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
