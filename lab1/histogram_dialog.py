import traceback
from PySide6.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem, QAbstractItemView,
    QVBoxLayout, QMessageBox, QDialog, QDialogButtonBox
)

import pandas as pd



from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HistogramDialog(QDialog):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.setWindowTitle("One-dimensional analysis — Histograms")
        self.resize(600, 400)
        self.df = df

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Choose numeric columns to plot histograms (multi-select):"))
        self.cols_list = QListWidget()
        self.cols_list.setSelectionMode(QAbstractItemView.MultiSelection)

        numeric_cols = self._detect_numeric_columns()
        if not numeric_cols:
            layout.addWidget(QLabel("No numeric columns found in DataFrame."))
        else:
            for c in numeric_cols:
                item = QListWidgetItem(c)
                self.cols_list.addItem(item)
            layout.addWidget(self.cols_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _detect_numeric_columns(self):
        numeric = list(self.df.select_dtypes(include=["number"]).columns)
        return numeric

    def on_accept(self):
        selected = [it.text() for it in self.cols_list.selectedItems()]
        if not selected:
            QMessageBox.information(self, "No selection", "Select at least one numeric column.")
            return
        try:
            self.plot_histograms(selected)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Plot error", f"Error while plotting:\n{e}\n\n{tb}")



    def plot_histograms(self, cols: list):
        print("Plotting ...")

        self.hist_window = QDialog(self)
        self.hist_window.setWindowTitle("Histograms — " + ", ".join(cols))
        self.hist_window.setModal(False)

        v = QVBoxLayout(self.hist_window)

        fig = Figure(figsize=(7, 3 * len(cols)))
        canvas = FigureCanvas(fig)
        v.addWidget(canvas)

        for i, col in enumerate(cols, start=1):
            ax = fig.add_subplot(len(cols), 1, i)
            data = self.df[col].dropna()

            if data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(col)
                continue

            if pd.api.types.is_integer_dtype(data) and data.nunique() < 20:
                bins = range(int(data.min()), int(data.max()) + 2)
            else:
                bins = 30

            ax.hist(data, bins=bins)
            ax.set_title(f"{col} (n={len(data)})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

        fig.tight_layout()
        canvas.draw()

        self.hist_window.resize(900, max(400, 280 * len(cols)))
        self.hist_window.show()
        self.hist_window.raise_()
        self.hist_window.activateWindow()
