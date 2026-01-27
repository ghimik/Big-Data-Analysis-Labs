import traceback
from PySide6.QtWidgets import (
    QLabel, QHBoxLayout, QVBoxLayout, QMessageBox, QCheckBox, QDialog,
    QDialogButtonBox, QComboBox, QSpinBox
)
from PySide6.QtCore import Qt

import pandas as pd
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MultivariateDialog(QDialog):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.setWindowTitle("Multivariate analysis — Scatter")
        self.resize(520, 260)
        self.df = df

        layout = QVBoxLayout(self)

        self.numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        self.categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        row_xy = QHBoxLayout()
        row_xy.addWidget(QLabel("X (numeric):"))
        self.x_cb = QComboBox()
        self.x_cb.addItems(self.numeric_cols)
        row_xy.addWidget(self.x_cb)

        row_xy.addWidget(QLabel("Y (numeric):"))
        self.y_cb = QComboBox()
        self.y_cb.addItems(self.numeric_cols)
        row_xy.addWidget(self.y_cb)
        layout.addLayout(row_xy)

        row_hue = QHBoxLayout()
        row_hue.addWidget(QLabel("Hue (categorical):"))
        self.hue_cb = QComboBox()
        self.hue_cb.addItem("— None —")
        self.hue_cb.addItems(self.categorical_cols)
        row_hue.addWidget(self.hue_cb)
        layout.addLayout(row_hue)

        options_row = QHBoxLayout()
        self.regress_cb = QCheckBox("Show regression line (linear)")
        options_row.addWidget(self.regress_cb)

        options_row.addWidget(QLabel("Point size:"))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 500)
        self.scale_spin.setValue(30)
        options_row.addWidget(self.scale_spin)
        layout.addLayout(options_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_accept(self):
        x = self.x_cb.currentText()
        y = self.y_cb.currentText()
        hue = self.hue_cb.currentText()
        regress = self.regress_cb.isChecked()
        scale = self.scale_spin.value()

        if not x or not y:
            QMessageBox.information(self, "Select", "Choose X and Y numeric columns.")
            return

        if hue == "— None —":
            hue = None

        try:
            self.plot_multivariate(x, y, hue, regress, scale)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Plot error", f"{e}\n\n{tb}")

    def plot_multivariate(self, x, y, hue=None, regress=False, scale=30):
        parent = self.parent()

        plot_win = QDialog(parent)
        plot_win.setWindowTitle(f"Scatter: {x} vs {y}")
        plot_win.setModal(False)
        plot_win.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        v = QVBoxLayout(plot_win)

        fig = Figure(figsize=(7, 5))
        canvas = FigureCanvas(fig)
        v.addWidget(canvas)

        ax = fig.add_subplot(1, 1, 1)

        cols = [x, y] + ([hue] if hue else [])
        df = self.df[cols].dropna()

        if hue:
            for cat, g in df.groupby(hue):
                ax.scatter(
                    g[x],
                    g[y],
                    s=scale,
                    alpha=0.7,
                    label=str(cat)
                )
            ax.legend(title=hue)
        else:
            ax.scatter(df[x], df[y], s=scale, alpha=0.7)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x} (n={len(df)})")

        if regress and len(df) > 1:
            coeffs = np.polyfit(df[x], df[y], 1)
            xs = np.linspace(df[x].min(), df[x].max(), 100)
            ys = np.polyval(coeffs, xs)
            ax.plot(xs, ys, linestyle="--", linewidth=1.5)

        fig.tight_layout()
        canvas.draw()

        # защита от GC
        if not hasattr(parent, "_plot_windows"):
            parent._plot_windows = []
        parent._plot_windows.append(plot_win)

        plot_win.resize(900, 600)
        plot_win.show()
        plot_win.raise_()
        plot_win.activateWindow()
