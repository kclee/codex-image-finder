"""Replaceable PySide6 presentation layer."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QProcess,
    QThread,
    QTimer,
    Qt,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis_queue import AnalysisQueue
from .analysis_specs import MOBILE_SUBTITLE_SPEC
from .catalog import Catalog
from .domain import SearchResult
from .review import ocr_review_reason
from .review_store import ReviewStore
from .text_search import to_traditional


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{max(1, round(seconds))}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


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
            remaining_ids = deque(batch_ids)
            while remaining_ids and not self.pause_requested.is_set():
                image_id = remaining_ids.popleft()
                job = queue.claim_next(run_id, [image_id])
                if job is None:
                    continue
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
    def __init__(self, catalog: Catalog, auto_scan: bool = False) -> None:
        super().__init__()
        self.catalog = catalog
        self.review_store = ReviewStore(
            self.catalog.project_root / "user-data" / "review-state.sqlite3"
        )
        self.current_group: str | None = None
        self.current_result: SearchResult | None = None
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.scan_is_automatic = False
        self.ocr_thread: QThread | None = None
        self.ocr_worker: OcrWorker | None = None
        self.ocr_batch_size = 10
        self.next_ocr_batch_ids: list[str] = []
        self.eligible_ocr_total = 0
        self.active_ocr_target = 0
        self.active_ocr_started_at = 0.0
        self.active_ocr_is_all = False
        self.close_after_ocr = False
        self.setWindowTitle("Image Finder — Prototype")
        self.resize(1280, 780)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search subtitle text, OCR text, or filename…")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self.refresh_results)

        self.review_filter_selector = QComboBox()
        self.review_filter_selector.addItem("Review: any status", None)
        self.review_filter_selector.addItem(
            "Review: needs correction", "needs_correction"
        )
        self.review_filter_selector.addItem("Review: not relevant", "not_relevant")
        self.review_filter_selector.addItem("Review: accepted OCR", "accepted")
        self.review_filter_selector.currentIndexChanged.connect(
            self.change_review_filter
        )

        self.scan_button = QPushButton("Add / scan folder…")
        self.scan_button.clicked.connect(self.choose_library)
        search_row = QWidget()
        search_layout = QHBoxLayout(search_row)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self.search_box, 1)
        search_layout.addWidget(self.review_filter_selector)
        search_layout.addWidget(self.scan_button)

        self.ocr_button = QPushButton(f"OCR next {self.ocr_batch_size}")
        self.ocr_button.setToolTip(
            "Process the next unprocessed images in the currently visible gallery"
        )
        self.ocr_button.clicked.connect(self.start_ocr_batch)
        self.ocr_all_button = QPushButton("OCR all remaining")
        self.ocr_all_button.setToolTip(
            "Process every unprocessed image in the currently visible gallery"
        )
        self.ocr_all_button.clicked.connect(self.start_all_remaining_ocr)
        self.pause_ocr_button = QPushButton("Pause OCR")
        self.pause_ocr_button.setEnabled(False)
        self.pause_ocr_button.clicked.connect(self.pause_ocr)
        self.retry_ocr_button = QPushButton("Retry failed")
        self.retry_ocr_button.clicked.connect(self.retry_failed_ocr)
        self.recent_ocr_button = QPushButton("Last OCR batch")
        self.recent_ocr_button.setCheckable(True)
        self.recent_ocr_button.toggled.connect(lambda _checked: self.refresh_results())
        self.review_ocr_button = QPushButton("Needs review")
        self.review_ocr_button.setCheckable(True)
        self.review_ocr_button.toggled.connect(self.change_needs_review_filter)
        self.ocr_history_button = QPushButton("OCR history…")
        self.ocr_history_button.clicked.connect(self.show_ocr_history)
        self.ocr_status = QLabel()
        self.ocr_status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        run_controls = QWidget()
        run_layout = QHBoxLayout(run_controls)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.addWidget(self.ocr_button)
        run_layout.addWidget(self.ocr_all_button)
        run_layout.addWidget(self.pause_ocr_button)
        run_layout.addWidget(self.retry_ocr_button)
        run_layout.addWidget(self.ocr_status, 1)
        view_controls = QWidget()
        view_layout = QHBoxLayout(view_controls)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.recent_ocr_button)
        view_layout.addWidget(self.review_ocr_button)
        view_layout.addWidget(self.ocr_history_button)
        view_layout.addStretch(1)
        ocr_row = QWidget()
        ocr_layout = QVBoxLayout(ocr_row)
        ocr_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(run_controls)
        ocr_layout.addWidget(view_controls)

        self.ocr_preview_label = QLabel("Next to OCR")
        self.batch_size_selector = QComboBox()
        for size in (10, 25, 50, 100):
            self.batch_size_selector.addItem(str(size), size)
        self.batch_size_selector.currentIndexChanged.connect(
            self.change_ocr_batch_size
        )
        preview_header = QWidget()
        preview_header_layout = QHBoxLayout(preview_header)
        preview_header_layout.setContentsMargins(0, 0, 0, 0)
        preview_header_layout.addWidget(self.ocr_preview_label, 1)
        preview_header_layout.addWidget(QLabel("Batch size:"))
        preview_header_layout.addWidget(self.batch_size_selector)
        self.ocr_estimate_label = QLabel()
        self.ocr_estimate_label.setWordWrap(True)
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
        center_layout.addWidget(preview_header)
        center_layout.addWidget(self.ocr_preview)
        center_layout.addWidget(self.ocr_estimate_label)
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
        self.review_status = QLabel("Review status: select a processed image")
        self.review_status.setWordWrap(True)
        self.accept_review_button = QPushButton("Accept OCR")
        self.accept_review_button.clicked.connect(
            lambda: self.set_review_decision("accepted")
        )
        self.correct_review_button = QPushButton("Needs correction")
        self.correct_review_button.clicked.connect(
            lambda: self.set_review_decision("needs_correction")
        )
        self.irrelevant_review_button = QPushButton("Not relevant")
        self.irrelevant_review_button.clicked.connect(
            lambda: self.set_review_decision("not_relevant")
        )
        self.clear_review_button = QPushButton("Clear decision")
        self.clear_review_button.clicked.connect(self.clear_review_decision)
        review_actions = QWidget()
        review_actions_layout = QGridLayout(review_actions)
        review_actions_layout.setContentsMargins(0, 0, 0, 0)
        review_actions_layout.addWidget(self.accept_review_button, 0, 0)
        review_actions_layout.addWidget(self.correct_review_button, 0, 1)
        review_actions_layout.addWidget(self.irrelevant_review_button, 1, 0)
        review_actions_layout.addWidget(self.clear_review_button, 1, 1)
        self.copy_subtitle_button = QPushButton("Copy subtitle text")
        self.copy_subtitle_button.setToolTip(
            "Copy the recognized subtitle converted to Traditional Chinese"
        )
        self.copy_subtitle_button.setEnabled(False)
        self.copy_subtitle_button.clicked.connect(self.copy_subtitle_text)
        self.open_folder_button = QPushButton("Open containing folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self.open_containing_folder)
        file_actions = QWidget()
        file_actions_layout = QHBoxLayout(file_actions)
        file_actions_layout.setContentsMargins(0, 0, 0, 0)
        file_actions_layout.addWidget(self.copy_subtitle_button)
        file_actions_layout.addWidget(self.open_folder_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.preview, 1)
        right_layout.addWidget(self.subtitle)
        right_layout.addWidget(self.analysis_selector)
        right_layout.addWidget(self.analysis_details)
        right_layout.addWidget(self.details)
        right_layout.addWidget(self.review_status)
        right_layout.addWidget(review_actions)
        right_layout.addWidget(file_actions)

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
        if auto_scan:
            QTimer.singleShot(0, self.start_startup_scan)

    def change_needs_review_filter(self, checked: bool) -> None:
        if checked and self.review_filter_selector.currentData() is not None:
            self.review_filter_selector.blockSignals(True)
            self.review_filter_selector.setCurrentIndex(0)
            self.review_filter_selector.blockSignals(False)
        self.refresh_results()

    def change_review_filter(self, _index: int) -> None:
        if self.review_filter_selector.currentData() is not None:
            self.review_ocr_button.blockSignals(True)
            self.review_ocr_button.setChecked(False)
            self.review_ocr_button.blockSignals(False)
        self.refresh_results()

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

    def start_startup_scan(self) -> None:
        roots = self.catalog.library_roots()
        if not roots:
            self.statusBar().showMessage(
                "No image library selected yet · use Add / scan folder…"
            )
            return
        library_root = roots[-1]
        if not library_root.is_dir():
            self.statusBar().showMessage(
                f"Saved image library is unavailable · {library_root}"
            )
            return
        self.start_scan(library_root, automatic=True)

    def start_scan(self, library_root: Path, automatic: bool = False) -> None:
        if self.scan_thread and self.scan_thread.isRunning():
            return
        self.scan_is_automatic = automatic
        self.scan_button.setEnabled(False)
        self.scan_button.setText(
            "Checking library…" if automatic else "Scanning folder…"
        )
        self.ocr_button.setEnabled(False)
        self.ocr_all_button.setEnabled(False)
        self.batch_size_selector.setEnabled(False)
        self.retry_ocr_button.setEnabled(False)
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setFormat("Scanning library…")
        self.scan_progress.show()
        action = "Checking for new or changed images" if automatic else "Scanning"
        self.statusBar().showMessage(f"{action} · {library_root}")

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
        self.scan_button.setText("Add / scan folder…")
        self.ocr_button.setEnabled(True)
        self.ocr_all_button.setEnabled(True)
        self.batch_size_selector.setEnabled(True)
        self.retry_ocr_button.setEnabled(True)
        self.populate_groups()
        self.refresh_results()
        prefix = (
            "Background library check complete"
            if self.scan_is_automatic
            else "Scan complete"
        )
        self.statusBar().showMessage(
            f"{prefix} · {summary.discovered:,} images · "
            f"{summary.hashed:,} hashed · {summary.unchanged:,} unchanged · "
            f"{summary.missing_marked:,} missing · {summary.errors:,} errors"
        )
        self.scan_is_automatic = False
        self.scan_worker = None
        self.scan_thread = None
        self.refresh_ocr_status()
        self.refresh_ocr_preview()

    def fail_scan(self, message: str) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.scan_button.setText("Add / scan folder…")
        self.ocr_button.setEnabled(True)
        self.ocr_all_button.setEnabled(True)
        self.batch_size_selector.setEnabled(True)
        self.retry_ocr_button.setEnabled(True)
        automatic = self.scan_is_automatic
        self.scan_is_automatic = False
        if automatic:
            self.statusBar().showMessage(f"Background library check failed · {message}")
        else:
            self.statusBar().showMessage("Scan failed")
            QMessageBox.critical(self, "Scan failed", message)
        self.scan_worker = None
        self.scan_thread = None
        self.refresh_ocr_status()
        self.refresh_ocr_preview()

    def refresh_ocr_status(self) -> None:
        counts = AnalysisQueue(self.catalog.connection).counts(MOBILE_SUBTITLE_SPEC.run_id)
        self.ocr_status.setText(
            f"OCR: {counts['succeeded']:,} complete · {counts['pending']:,} pending · "
            f"{counts['failed']:,} failed"
        )
        self.retry_ocr_button.setEnabled(
            counts["failed"] > 0 and not (self.ocr_thread and self.ocr_thread.isRunning())
        )

    def change_ocr_batch_size(self, _index: int) -> None:
        selected = self.batch_size_selector.currentData()
        if selected is None:
            return
        self.ocr_batch_size = int(selected)
        self.ocr_button.setText(f"OCR next {self.ocr_batch_size}")
        self.refresh_ocr_preview()

    def refresh_ocr_preview(self) -> None:
        if self.ocr_thread and self.ocr_thread.isRunning():
            return
        ordered_results: dict[str, SearchResult] = {}
        for result in self.gallery_model.results:
            ordered_results.setdefault(result.image_id, result)
        queue = AnalysisQueue(self.catalog.connection)
        batch_ids, total = queue.preview_ordered_batch(
            MOBILE_SUBTITLE_SPEC.run_id,
            list(ordered_results),
            self.ocr_batch_size,
        )
        self.eligible_ocr_total = total
        self.next_ocr_batch_ids = batch_ids
        visible_preview_ids = batch_ids[:10]
        self.ocr_preview_model.replace(
            [ordered_results[image_id] for image_id in visible_preview_ids]
        )
        batches = (total + self.ocr_batch_size - 1) // self.ocr_batch_size
        timing = self.catalog.analysis_timing(MOBILE_SUBTITLE_SPEC.run_id)
        if total:
            self.ocr_preview.show()
            if len(batch_ids) > len(visible_preview_ids):
                preview_text = (
                    f"showing first {len(visible_preview_ids)} thumbnails · "
                    f"next batch {len(batch_ids)} of {total:,} eligible"
                )
            else:
                preview_text = (
                    f"showing {len(visible_preview_ids)} of {total:,} eligible"
                )
            self.ocr_preview_label.setText(
                f"Next to OCR · {preview_text} in this view · about {batches:,} "
                f"batch{'es' if batches != 1 else ''}"
            )
            self.ocr_button.setEnabled(
                not (self.scan_thread and self.scan_thread.isRunning())
            )
            self.ocr_all_button.setText(
                f"OCR all remaining ({total:,})"
            )
            self.ocr_all_button.setEnabled(
                not (self.scan_thread and self.scan_thread.isRunning())
            )
            if timing:
                next_seconds = timing.median_seconds * len(batch_ids)
                total_seconds = timing.median_seconds * total
                self.ocr_estimate_label.setText(
                    f"Typical OCR: {timing.median_seconds:.2f}s/image "
                    f"(median of {timing.sample_count} completed images)\n"
                    f"Selected batch: {len(batch_ids)} images · about "
                    f"{_format_duration(next_seconds)} OCR + one model startup · "
                    f"Remaining: {total:,} images · about "
                    f"{_format_duration(total_seconds)} OCR across {batches:,} "
                    f"batch{'es' if batches != 1 else ''}"
                )
            else:
                self.ocr_estimate_label.setText(
                    "Timing estimate available after the first measured OCR result"
                )
        else:
            self.next_ocr_batch_ids = []
            self.ocr_preview.hide()
            self.ocr_preview_label.setText("Next to OCR · no eligible images in this view")
            self.ocr_estimate_label.setText("No remaining OCR estimate for this view")
            self.ocr_button.setEnabled(False)
            self.ocr_all_button.setText("OCR all remaining")
            self.ocr_all_button.setEnabled(False)

    def show_ocr_history(self) -> None:
        summaries = self.catalog.analysis_batch_summaries(
            MOBILE_SUBTITLE_SPEC.run_id, limit=12
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Recent OCR batches")
        dialog.resize(820, 390)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "Recent persisted queue groups. OCR time is measured inference only and "
            "does not include model loading."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(len(summaries), 8)
        table.setHorizontalHeaderLabels(
            [
                "Queued",
                "Total",
                "Complete",
                "Pending",
                "Running",
                "Failed",
                "Skipped",
                "OCR time",
            ]
        )
        for row_index, summary in enumerate(summaries):
            values = [
                summary.queued_at[:19].replace("T", " "),
                str(summary.total),
                str(summary.succeeded),
                str(summary.pending),
                str(summary.running),
                str(summary.failed),
                str(summary.skipped),
                _format_duration(summary.inference_seconds)
                if summary.inference_seconds
                else "—",
            ]
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def start_ocr_batch(self) -> None:
        self.start_ocr(self.ocr_batch_size, all_remaining=False)

    def start_all_remaining_ocr(self) -> None:
        if self.eligible_ocr_total:
            self.start_ocr(self.eligible_ocr_total, all_remaining=True)

    def start_ocr(self, batch_size: int, all_remaining: bool) -> None:
        if self.ocr_thread and self.ocr_thread.isRunning():
            return
        ordered_image_ids = list(
            dict.fromkeys(result.image_id for result in self.gallery_model.results)
        )
        batch_ids, _total = AnalysisQueue(
            self.catalog.connection
        ).preview_ordered_batch(
            MOBILE_SUBTITLE_SPEC.run_id,
            ordered_image_ids,
            batch_size,
        )
        if not batch_ids:
            self.refresh_ocr_preview()
            return
        self.scan_button.setEnabled(False)
        self.ocr_button.setEnabled(False)
        self.ocr_all_button.setEnabled(False)
        self.batch_size_selector.setEnabled(False)
        self.retry_ocr_button.setEnabled(False)
        self.pause_ocr_button.setEnabled(True)
        self.active_ocr_target = len(batch_ids)
        self.active_ocr_started_at = time.monotonic()
        self.active_ocr_is_all = all_remaining
        self.scan_progress.setRange(0, self.active_ocr_target)
        self.scan_progress.setFormat(
            "OCR all remaining: %v / %m images (%p%)"
            if all_remaining
            else "Current batch: %v / %m images (%p%)"
        )
        self.scan_progress.setValue(0)
        self.scan_progress.show()
        self.statusBar().showMessage("Preparing local OCR models…")
        self.ocr_preview_label.setText(
            f"OCR all remaining · {self.active_ocr_target:,} image(s)"
            if all_remaining
            else f"Current OCR batch · {self.active_ocr_target} image(s)"
        )

        self.ocr_thread = QThread(self)
        self.ocr_worker = OcrWorker(
            self.catalog.database_path,
            self.catalog.project_root,
            len(batch_ids),
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
        scope = "all remaining" if self.active_ocr_is_all else "batch"
        self.statusBar().showMessage(
            f"OCR {scope} {processed:,}/{self.active_ocr_target:,} · {filename}"
        )
        elapsed = max(0.0, time.monotonic() - self.active_ocr_started_at)
        if processed > 0:
            remaining = max(0, self.active_ocr_target - processed)
            estimated_remaining = elapsed / processed * remaining
            self.ocr_estimate_label.setText(
                f"Running: {processed:,} / {self.active_ocr_target:,} images · "
                f"elapsed {_format_duration(elapsed)} · estimated "
                f"{_format_duration(estimated_remaining)} remaining"
            )
        self.refresh_ocr_status()

    def finish_ocr(self, summary: object) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.ocr_button.setEnabled(True)
        self.ocr_all_button.setEnabled(True)
        self.batch_size_selector.setEnabled(True)
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
        self.refresh_ocr_preview()
        if self.close_after_ocr:
            QTimer.singleShot(0, self.close)

    def fail_ocr(self, message: str) -> None:
        self.scan_progress.hide()
        self.scan_button.setEnabled(True)
        self.ocr_button.setEnabled(True)
        self.ocr_all_button.setEnabled(True)
        self.batch_size_selector.setEnabled(True)
        self.pause_ocr_button.setEnabled(False)
        self.refresh_ocr_status()
        self.statusBar().showMessage("OCR worker failed")
        QMessageBox.critical(self, "OCR worker failed", message)
        self.ocr_worker = None
        self.ocr_thread = None
        self.refresh_ocr_preview()
        if self.close_after_ocr:
            QTimer.singleShot(0, self.close)

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
        decisions = self.review_store.decisions_for_run(MOBILE_SUBTITLE_SPEC.run_id)
        review_results = [
            result
            for result in results
            if result.analysis_run_id == MOBILE_SUBTITLE_SPEC.run_id
            and ocr_review_reason(result.subtitle_text, result.confidence) is not None
            and result.image_id not in decisions
        ]
        self.review_ocr_button.setText(f"Needs review ({len(review_results):,})")
        if self.review_ocr_button.isChecked():
            results = review_results
        elif decision_filter := self.review_filter_selector.currentData():
            results = [
                result
                for result in results
                if decisions.get(result.image_id) == decision_filter
            ]
        self.gallery_model.replace(results)
        self.count_label.setText(
            f"Gallery: {len(results):,} image{'s' if len(results) != 1 else ''}"
        )
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
            self.refresh_review_controls(False)
            self.copy_subtitle_button.setEnabled(False)
            self.open_folder_button.setEnabled(False)
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
        self.refresh_review_controls(
            any(record.run_id == MOBILE_SUBTITLE_SPEC.run_id for record in history)
        )
        self.open_folder_button.setEnabled(result.absolute_path.is_file())

    def refresh_review_controls(self, has_current_ocr: bool) -> None:
        buttons = (
            self.accept_review_button,
            self.correct_review_button,
            self.irrelevant_review_button,
        )
        for button in buttons:
            button.setEnabled(has_current_ocr)
        if self.current_result is None:
            self.review_status.setText("Review status: select a processed image")
            self.clear_review_button.setEnabled(False)
            return
        if not has_current_ocr:
            self.review_status.setText("Review status: current OCR required")
            self.clear_review_button.setEnabled(False)
            return
        decision = self.review_store.decision(
            self.current_result.image_id, MOBILE_SUBTITLE_SPEC.run_id
        )
        labels = {
            "accepted": "Accepted OCR",
            "needs_correction": "Needs correction",
            "not_relevant": "Not relevant",
        }
        self.review_status.setText(
            f"Review status: {labels.get(decision, 'Not reviewed')}"
        )
        self.clear_review_button.setEnabled(decision is not None)

    def set_review_decision(self, decision: str) -> None:
        if self.current_result is None:
            return
        current_row = max(0, self.gallery.currentIndex().row())
        self.review_store.set_decision(
            self.current_result.image_id, MOBILE_SUBTITLE_SPEC.run_id, decision
        )
        self.refresh_results()
        self.select_review_row(current_row)

    def clear_review_decision(self) -> None:
        if self.current_result is None:
            return
        current_row = max(0, self.gallery.currentIndex().row())
        self.review_store.clear_decision(
            self.current_result.image_id, MOBILE_SUBTITLE_SPEC.run_id
        )
        self.refresh_results()
        self.select_review_row(current_row)

    def select_review_row(self, row: int) -> None:
        if self.gallery_model.rowCount() == 0:
            self.gallery.setCurrentIndex(QModelIndex())
            return
        next_index = self.gallery_model.index(
            min(row, self.gallery_model.rowCount() - 1), 0
        )
        self.gallery.setCurrentIndex(next_index)

    def show_analysis_version(self, index: int) -> None:
        record = self.analysis_selector.itemData(index) if index >= 0 else None
        if record is None:
            self.subtitle.setText("(No recognized text)")
            self.analysis_details.setText("Current OCR version: not processed")
            self.copy_subtitle_button.setEnabled(False)
            return
        self.subtitle.setText(record.display_text)
        self.copy_subtitle_button.setEnabled(bool(record.subtitle_text.strip()))
        confidence = "—" if record.confidence is None else f"{record.confidence:.1%}"
        current = (
            "Current OCR version"
            if record.run_id == MOBILE_SUBTITLE_SPEC.run_id
            else "Earlier analysis version"
        )
        review_reason = (
            ocr_review_reason(record.subtitle_text, record.confidence)
            if record.run_id == MOBILE_SUBTITLE_SPEC.run_id
            else None
        )
        review_line = f"\nReview suggested: {review_reason}" if review_reason else ""
        self.analysis_details.setText(
            f"{current} · confidence {confidence}\n"
            f"{record.engine_name} {record.engine_version} · {record.model_name}\n"
            f"Pipeline: {record.pipeline_version}\nCompleted: {record.created_at}"
            f"{review_line}"
        )

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self.current_result:
            current = self.gallery.currentIndex()
            if current.isValid():
                self.show_result(current, QModelIndex())

    def copy_subtitle_text(self) -> None:
        record = self.analysis_selector.currentData()
        if record is None or not record.subtitle_text.strip():
            return
        copied_text = to_traditional(record.subtitle_text.strip())
        QApplication.clipboard().setText(copied_text)
        self.statusBar().showMessage(
            "Traditional Chinese subtitle copied to the clipboard"
        )

    def open_containing_folder(self) -> None:
        if self.current_result is None:
            return
        path = self.current_result.absolute_path
        if not path.is_file():
            return
        if sys.platform == "win32":
            QProcess.startDetached("explorer.exe", ["/select,", str(path)])
        elif sys.platform == "darwin":
            QProcess.startDetached("open", ["-R", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        self.statusBar().showMessage("Opened the containing folder")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.ocr_thread and self.ocr_thread.isRunning():
            self.close_after_ocr = True
            self.pause_ocr()
            self.statusBar().showMessage(
                "Closing after the current OCR image is safely stored…"
            )
            event.ignore()
            return
        self.review_store.close()
        super().closeEvent(event)


def run(catalog: Catalog, smoke_test: bool = False) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Image Finder")
    window = MainWindow(catalog, auto_scan=not smoke_test)
    window.show()
    if smoke_test:
        QTimer.singleShot(250, app.quit)
    return app.exec()
