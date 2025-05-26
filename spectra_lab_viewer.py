#!/usr/bin/env python3
"""SpectraLabViewer – full release
=================================
A turnkey PyQt5 + Matplotlib programme to visualise, analyse and simulate
one‑dimensional spectra stored in Horiba **.LAB** files. Designed for
routine use at the bench: drag‑and‑drop, multi‑file overlay, rich plot
customisation, log/√ axes, ROI peak fitting and export of both plots and
simulated 2‑D spectral images.

Dependencies
------------
• Python ≥ 3.9  • PyQt5  • Matplotlib  • NumPy  • SciPy  • Pillow

Install everything with ::

    pip install pyqt5 matplotlib numpy scipy pillow

Run with ::

    python spectra_lab_viewer.py
"""
from __future__ import annotations

import os, re, sys, math, contextlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.scale as mscale

# --------------------------------------------------------------------------
#                         Custom square-root scale
# --------------------------------------------------------------------------
class SqrtScale(mscale.ScaleBase):
    name = "sqrt"

    def __init__(self, axis, **kwargs):
        super().__init__()

    def get_transform(self):
        class SqrtTransform(mscale.Transform):
            input_dims = output_dims = 1
            def transform_non_affine(self, a):
                return np.sqrt(a)
            def inverted(self):
                return InvSqrtTransform()
        class InvSqrtTransform(mscale.Transform):
            input_dims = output_dims = 1
            def transform_non_affine(self, a):
                return np.square(a)
            def inverted(self):
                return SqrtTransform()
        return SqrtTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mscale.LogLocator(base=10, numticks=12))
        axis.set_major_formatter(mscale.LogFormatter())

mscale.register_scale(SqrtScale)

# --------------------------------------------------------------------------
#                        Spectrum data model
# --------------------------------------------------------------------------
class Spectrum:
    """Container for one spectrum (x, y) plus metadata."""
    def __init__(self, name: str, x: NDArray, y: NDArray, path: str):
        self.name = name
        self.x = x
        self.y = y
        self.path = path
        self.visible = True
        self.line = None

    @staticmethod
    def gauss(x, amp, cen, sig, off):
        return amp * np.exp(-((x - cen)**2) / (2*sig**2)) + off

    def fit_roi(self, x0: float, x1: float):
        sel = (self.x >= x0) & (self.x <= x1)
        if sel.sum() < 5:
            raise RuntimeError("ROI too small")
        xr, yr = self.x[sel], self.y[sel]
        p0 = [yr.max() - yr.min(), xr[np.argmax(yr)], (x1-x0)/6, yr.min()]
        popt, pcov = curve_fit(self.gauss, xr, yr, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        keys = ("Amplitude", "Center", "Sigma", "Offset")
        return {k: (v, e) for k, v, e in zip(keys, popt, perr)}

# --------------------------------------------------------------------------
#                            .LAB file parsing
# --------------------------------------------------------------------------
_vec_re = re.compile(
    r"\[vecteur\]\s*(.*?)"          # header block
    r"points\s*=\s*(\d+).*?"        # point count
    r"table\s*{(.*?)}",             # data block
    re.S | re.I
)
_vec_index_re = re.compile(
    r"\[vecteur\s*\d+\]\s*(.*?)"
    r"points\s*=\s*(\d+).*?"
    r"table\s*{(.*?)}",
    re.S | re.I
)
_name_re = re.compile(r"nom\s*=\s*\"([^\"]*)\"", re.I)

def _numbers(text: str) -> NDArray:
    return np.fromstring(re.sub(r"[^0-9eE.+-]", " ", text), sep=" ")

def parse_lab(path: str) -> list[tuple[str, NDArray]]:
    """Return list of (name, array) for every vector in the .lab file."""
    with open(path, "rb") as fh:
        txt = fh.read().decode("utf-16-le", errors="ignore")
    vectors: list[tuple[str, NDArray]] = []
    for pattern in (_vec_re, _vec_index_re):
        for hdr, npts, body in pattern.findall(txt):
            m = _name_re.search(hdr)
            name = m.group(1) if m else f"vec{len(vectors)}"
            data = _numbers(body)
            if data.size:
                vectors.append((name, data))
    return vectors


# --------------------------------------------------------------------------
#                       Heuristic .LAB loader
# --------------------------------------------------------------------------
def read_lab(path: str) -> list[Spectrum]:
    """
    Heuristic loader: uses parse_lab to get all vectors, then picks an X-axis
    (first monotonic or named 'wave', etc.) and returns Spectrum objects
    for every other vector.
    """
    vectors = parse_lab(path)
    if not vectors:
        return []
    # pick X-axis by name keyword
    x = None
    for name, data in vectors:
        if any(k in name.lower() for k in ("wave", "lambda", "wavelength", "x")):
            x = data
            break
    # fallback: first strictly increasing vector
    if x is None:
        for _, data in vectors:
            if np.all(np.diff(data) > 0):
                x = data
                break
    # final fallback: index array
    if x is None:
        x = np.arange(len(vectors[0][1]))
    specs: list[Spectrum] = []
    for name, data in vectors:
        if data is x:
            continue
        specs.append(Spectrum(name, x, data, path))
    return specs

# --------------------------------------------------------------------------
#                       Matplotlib canvas widget
# --------------------------------------------------------------------------
class PlotCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(constrained_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.series: list[Spectrum] = []

    def add(self, spec: Spectrum):
        line, = self.ax.plot(spec.x, spec.y, label=spec.name, lw=1.2, picker=5)
        spec.line = line
        self.series.append(spec)
        self.ax.legend()
        self.draw_idle()

    def update_visibility(self, spec: Spectrum):
        if spec.line:
            spec.line.set_visible(spec.visible)
            self.draw_idle()

    def set_yscale(self, scale: str):
        self.ax.set_yscale(scale)
        self.draw_idle()

    def clear(self):
        self.ax.clear()
        self.series.clear()
        self.draw_idle()

# --------------------------------------------------------------------------
#                   Series-import helper dialog
# --------------------------------------------------------------------------
class SeriesImportDialog(QtWidgets.QDialog):
    def __init__(self, file_path: str, vectors: list[tuple[str, NDArray]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import spectra – {Path(file_path).name}")
        self._vectors = vectors

        # X-axis selector
        x_box = QtWidgets.QGroupBox("Choose X-axis vector")
        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.addItems([n for n, _ in vectors])
        x_layout = QtWidgets.QVBoxLayout(x_box)
        x_layout.addWidget(self.x_combo)

        # Y-series checklist
        y_box = QtWidgets.QGroupBox("Y-series to import")
        self.y_list = QtWidgets.QListWidget()
        self.y_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for n, _ in vectors:
            item = QtWidgets.QListWidgetItem(n)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.y_list.addItem(item)
        y_layout = QtWidgets.QVBoxLayout(y_box)
        y_layout.addWidget(self.y_list)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(x_box)
        layout.addWidget(y_box)
        layout.addWidget(btns)

    def x_axis_name(self) -> str:
        return self.x_combo.currentText()

    def y_series_names(self) -> list[str]:
        return [
            self.y_list.item(i).text()
            for i in range(self.y_list.count())
            if self.y_list.item(i).checkState() == QtCore.Qt.Checked
        ]

# --------------------------------------------------------------------------
#                              Main window
# --------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    FILE_FILTER = "LAB spectra (*.lab);;All files (*)"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraLabViewer ⸱ 2025.05.22")
        self.resize(1000, 700)

        self.canvas = PlotCanvas()
        toolbar = NavigationToolbar(self.canvas, self)
        container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(container)
        vlayout.addWidget(toolbar)
        vlayout.addWidget(self.canvas)
        self.setCentralWidget(container)

        # Series dock
        self.series_model = QtGui.QStandardItemModel()
        listview = QtWidgets.QListView()
        listview.setModel(self.series_model)
        listview.clicked.connect(self._toggle_series)
        dock = QtWidgets.QDockWidget("Series", self)
        dock.setWidget(listview)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # Scale toolbar
        scale_tb = QtWidgets.QToolBar("Scale")
        scale_tb.addWidget(QtWidgets.QLabel("Y-axis:"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["linear", "log", "sqrt"])
        self.scale_combo.currentTextChanged.connect(self._scale_changed)
        scale_tb.addWidget(self.scale_combo)
        self.addToolBar(scale_tb)

        # Menu
        open_act = QtWidgets.QAction("&Open…", self, shortcut="Ctrl+O", triggered=self.open_dialog)
        save_act = QtWidgets.QAction("Save &plot as PNG…", self, triggered=self.save_plot)
        fit_act = QtWidgets.QAction("&Fit peak…", self, shortcut="F", triggered=self.start_roi_fit)
        sim_act = QtWidgets.QAction("&Simulate image…", self, triggered=self.simulate_image)
        quit_act = QtWidgets.QAction("Quit", self, shortcut="Ctrl+Q", triggered=self.close)

        m_file = self.menuBar().addMenu("&File")
        m_file.addAction(open_act)
        m_file.addSeparator()
        m_file.addAction(save_act)
        m_file.addSeparator()
        m_file.addAction(quit_act)
        m_analysis = self.menuBar().addMenu("&Analysis")
        m_analysis.addAction(fit_act)
        m_analysis.addAction(sim_act)

        self._span_selector: SpanSelector | None = None
        self._fit_dock: QtWidgets.QDockWidget | None = None

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.load_files(paths)

    def open_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open spectra", str(Path.home()), self.FILE_FILTER
        )
        self.load_files(paths)

    def load_files(self, paths: list[str]):
        any_loaded = False
        for p in paths:
            loaded = False
            # First attempt user-driven import dialog
            try:
                vectors = parse_lab(p)
                if vectors:
                    dlg = SeriesImportDialog(p, vectors, self)
                    if dlg.exec_() == QtWidgets.QDialog.Accepted:
                        x_name = dlg.x_axis_name()
                        y_names = dlg.y_series_names()
                        if y_names:
                            vecs = {n: d for n, d in vectors}
                            x_vec = vecs.get(x_name)
                            if x_vec is None:
                                raise RuntimeError("Chosen X axis not found")
                            for y in y_names:
                                y_vec = vecs.get(y)
                                if y_vec is None or y == x_name:
                                    continue
                                if y_vec.size != x_vec.size:
                                    QtWidgets.QMessageBox.warning(
                                        self, "Size mismatch",
                                        f"{y}: length {y_vec.size} vs X ({x_vec.size}); skipped."
                                    )
                                    continue
                                spec = Spectrum(y, x_vec, y_vec, p)
                                self.canvas.add(spec)
                                item = QtGui.QStandardItem(spec.name)
                                item.setCheckable(True)
                                item.setCheckState(QtCore.Qt.Checked)
                                item.setData(spec)
                                self.series_model.appendRow(item)
                            loaded = True
            except Exception:
                pass
            # Fallback heuristic loader
            if not loaded:
                try:
                    specs = read_lab(p)
                    if not specs:
                        raise RuntimeError("No spectra found")
                    for spec in specs:
                        self.canvas.add(spec)
                        item = QtGui.QStandardItem(spec.name)
                        item.setCheckable(True)
                        item.setCheckState(QtCore.Qt.Checked)
                        item.setData(spec)
                        self.series_model.appendRow(item)
                    loaded = True
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Load error",
                                                  f"{Path(p).name}:\n{exc}")
            if loaded:
                any_loaded = True
        if not any_loaded:
            QtWidgets.QMessageBox.information(self, "No spectra loaded", "No spectra were loaded from the selected file(s). Please check the file format or selection.")

    def _toggle_series(self, index: QtCore.QModelIndex):
        item = self.series_model.itemFromIndex(index)
        spec = item.data()
        spec.visible = not spec.visible
        item.setCheckState(QtCore.Qt.Checked if spec.visible else QtCore.Qt.Unchecked)
        self.canvas.update_visibility(spec)

    def _scale_changed(self, scale: str):
        self.canvas.set_yscale(scale)

    def save_plot(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plot", "plot.png", "PNG image (*.png)"
        )
        if fname:
            self.canvas.fig.savefig(fname, dpi=300)

    def start_roi_fit(self):
        if self._span_selector:
            self._span_selector.disconnect_events()
            self._span_selector = None
            self.statusBar().clearMessage()
            return
        self.statusBar().showMessage("Drag over a peak to fit (Esc to cancel)")
        self._span_selector = SpanSelector(
            self.canvas.ax, self._roi_selected, "horizontal", useblit=True, facecolor="tab:red"
        )

    def _roi_selected(self, x0: float, x1: float):
        self.statusBar().clearMessage()
        if self._span_selector:
            self._span_selector.disconnect_events()
            self._span_selector = None
        if x1 <= x0:
            return

        results = []
        for spec in self.canvas.series:
            if spec.visible:
                try:
                    res = spec.fit_roi(x0, x1)
                    results.append((spec.name, res))
                except Exception:
                    pass
        if not results:
            QtWidgets.QMessageBox.information(
                self, "Fit", "No visible spectra could be fitted in the selected region."
            )
            return

        table = QtWidgets.QTableWidget()
        keys = ("Amplitude", "Center", "Sigma", "Offset")
        headers = ("Series",) + keys
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(results))
        for row, (name, res) in enumerate(results):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            for col, key in enumerate(keys, 1):
                val, err = res[key]
                table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{val:.4g} ± {err:.2g}"))
        table.resizeColumnsToContents()
        if not self._fit_dock:
            self._fit_dock = QtWidgets.QDockWidget("Peak fit", self)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._fit_dock)
        self._fit_dock.setWidget(table)
        self._fit_dock.show()

    def simulate_image(self):
        if not self.canvas.series:
            QtWidgets.QMessageBox.information(self, "Simulate", "Load spectra first.")
            return
        maxlen = max(len(s.y) for s in self.canvas.series if s.visible)
        img = np.zeros((len(self.canvas.series), maxlen), float)
        for row, s in enumerate(self.canvas.series):
            if not s.visible:
                continue
            y = np.interp(np.linspace(0, 1, maxlen), np.linspace(0, 1, len(s.y)), s.y)
            img[row] = y
        img -= img.min()
        img /= img.max() or 1
        img8 = (img * 255).astype(np.uint8)
        im = Image.fromarray(img8, mode="L")
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", "spectra.png", "PNG image (*.png);;TIFF (*.tif)"
        )
        if fname:
            im.save(fname)

def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
