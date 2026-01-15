import sys
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QAbstractItemView,
    QHBoxLayout, QVBoxLayout, QMessageBox, QCheckBox, QTextEdit, QFileDialog,
    QTableView, QHeaderView, QSizePolicy, QDialog, QDialogButtonBox, 
    QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

import numpy as np

try:
    from config import db_config  
except Exception:
    db_config = None

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame() if df is None else df.reset_index(drop=True)

    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = pd.DataFrame() if df is None else df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self._df.empty else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            text = str(value)
            return text if len(text) < 200 else text[:200] + "..."
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        else:
            return section + 1


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


class MultivariateDialog(QDialog):
    def __init__(self, parent, df: pd.DataFrame):
        super().__init__(parent)
        self.setWindowTitle("Multivariate analysis — Scatter")
        self.resize(480, 220)
        self.df = df

        layout = QVBoxLayout()
        self.setLayout(layout)

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


        options_row = QHBoxLayout()
        self.regress_cb = QCheckBox("Show regression line (linear)")
        options_row.addWidget(self.regress_cb)
        options_row.addWidget(QLabel("Point scale factor:"))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 1000)
        self.scale_spin.setValue(20)
        options_row.addWidget(self.scale_spin)
        layout.addLayout(options_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_accept(self):
        x = self.x_cb.currentText()
        y = self.y_cb.currentText()
        regress = self.regress_cb.isChecked()
        scale = self.scale_spin.value()

        if not x or not y:
            QMessageBox.information(self, "Select", "Choose X and Y numeric columns.")
            return


        try:
            self.plot_multivariate(x, y, regress, scale)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Plot error", f"Error while plotting:\n{e}\n\n{tb}")

    def plot_multivariate(self, x, y, regress=False, scale=20):
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

        df = self.df[[x, y]].dropna()
        X = df[x].to_numpy()
        Y = df[y].to_numpy()

        ax.scatter(X, Y, s=30, alpha=0.7)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x} (n={len(df)})")

        if regress:
            coeffs = np.polyfit(X, Y, 1)
            xs = np.linspace(X.min(), X.max(), 100)
            ys = np.polyval(coeffs, xs)
            ax.plot(xs, ys, linestyle="--", linewidth=1.5)

        fig.tight_layout()
        canvas.draw()

        # ЕСЛИ ЭТО УБРАТЬ ТО НИЧЕГО НЕ ПОКАЖЕТСЯ!!!!!! ПРИВЕТ GC PYTHON !!!!!! ЛЮБЛЮ ОБОЖАЮ
        if not hasattr(parent, "_plot_windows"):
            parent._plot_windows = []
        parent._plot_windows.append(plot_win)

        plot_win.resize(900, 600)
        plot_win.show()
        plot_win.raise_()
        plot_win.activateWindow()




class DBFeatureSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature picker — Football analysis (with histograms)")
        self.resize(1200, 800)

        self.engine = None
        self.inspector = None
        self.tables = []
        self.table_columns = {}

        self._build_ui()
        self.current_df = pd.DataFrame()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        conn_box = QWidget()
        conn_layout = QGridLayout()
        conn_box.setLayout(conn_layout)

        row = 0
        conn_layout.addWidget(QLabel("<b>DB connection</b>"), row, 0, 1, 3)
        row += 1

        self.host_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.db_edit = QLineEdit()
        self.user_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.Password)

        if db_config is not None:
            try:
                self.host_edit.setText(db_config.host)
                self.port_edit.setText(str(db_config.port))
                self.db_edit.setText(db_config.database)
                self.user_edit.setText(db_config.user)
                self.pass_edit.setText(db_config.password)
            except Exception:
                pass

        conn_layout.addWidget(QLabel("Host:"), row, 0)
        conn_layout.addWidget(self.host_edit, row, 1)
        row += 1
        conn_layout.addWidget(QLabel("Port:"), row, 0)
        conn_layout.addWidget(self.port_edit, row, 1)
        row += 1
        conn_layout.addWidget(QLabel("Database:"), row, 0)
        conn_layout.addWidget(self.db_edit, row, 1)
        row += 1
        conn_layout.addWidget(QLabel("User:"), row, 0)
        conn_layout.addWidget(self.user_edit, row, 1)
        row += 1
        conn_layout.addWidget(QLabel("Password:"), row, 0)
        conn_layout.addWidget(self.pass_edit, row, 1)
        row += 1

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect)
        conn_layout.addWidget(self.connect_btn, row, 0, 1, 2)
        row += 1

        self.tables_list = QListWidget()
        self.tables_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tables_list.itemSelectionChanged.connect(self.on_table_selected)

        self.cols_list = QListWidget()
        self.cols_list.setSelectionMode(QAbstractItemView.MultiSelection)

        self.autojoin_check = QCheckBox("Apply automatic JOIN map (recommended)")
        self.autojoin_check.setChecked(True)

        self.add_cols_btn = QPushButton("Добавить выбранные колонки в итог")
        self.add_cols_btn.clicked.connect(self.add_selected_columns)

        left_vbox = QVBoxLayout()
        left_vbox.addWidget(QLabel("<b>Tables</b>"))
        left_vbox.addWidget(self.tables_list)
        left_vbox.addWidget(QLabel("<b>Columns (select + Add)</b>"))
        left_vbox.addWidget(self.cols_list)
        left_vbox.addWidget(self.autojoin_check)
        left_vbox.addWidget(self.add_cols_btn)

        features_box = QWidget()
        features_layout = QVBoxLayout()
        features_box.setLayout(features_layout)

        features_layout.addWidget(QLabel("<b>Selected features (final table)</b>"))
        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        features_layout.addWidget(self.features_text)

        self.build_btn = QPushButton("Build DataFrame")
        self.build_btn.clicked.connect(self.on_build_dataframe)
        features_layout.addWidget(self.build_btn)

        self.clear_btn = QPushButton("Clear selected features")
        self.clear_btn.clicked.connect(self.on_clear_features)
        features_layout.addWidget(self.clear_btn)


        self.hist_btn = QPushButton("One-dimensional analysis (histograms)")
        self.hist_btn.setEnabled(False)
        self.hist_btn.clicked.connect(self.on_histograms)
        features_layout.addWidget(self.hist_btn)

        self.multi_btn = QPushButton("Multivariate analysis (scatter)")
        self.multi_btn.setEnabled(False)
        self.multi_btn.clicked.connect(self.on_multivariate)
        features_layout.addWidget(self.multi_btn)


        self.save_csv_btn = QPushButton("Save CSV")
        self.save_csv_btn.clicked.connect(self.on_save_csv)
        self.save_csv_btn.setEnabled(False)
        features_layout.addWidget(self.save_csv_btn)

        self.table_view = QTableView()
        self.table_model = DataFrameModel()
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)

        left_widget = QWidget()
        left_widget.setLayout(left_vbox)

        grid.addWidget(conn_box, 0, 0)
        grid.addWidget(left_widget, 1, 0, 2, 1)
        grid.addWidget(features_box, 0, 1, 2, 1)
        grid.addWidget(self.table_view, 2, 1)
        grid.addWidget(QLabel("<b>Log</b>"), 3, 0)
        grid.addWidget(self.log_text, 3, 1)

        self.selected_cols = {}

        self.join_map = [
            ("matches", "STADIUM", "stadiums", "NAME"),
            ("matches", "HOME_TEAM", "teams", "TEAM_NAME"),
            ("matches", "AWAY_TEAM", "teams", "TEAM_NAME"),
            ("players", "TEAM", "teams", "TEAM_NAME"),
            ("goals", "MATCH_ID", "matches", "MATCH_ID"),
            ("goals", "PID", "players", "PLAYER_ID"),
        ]

    def log(self, *parts):
        self.log_text.append(" ".join(map(str, parts)))

    def on_multivariate(self):
        if self.current_df is None or self.current_df.empty:
            QMessageBox.information(self, "No data", "Сначала соберите DataFrame (Build DataFrame).")
            return
        dlg = MultivariateDialog(self, self.current_df)
        dlg.exec() 


    def on_connect(self):
        host = self.host_edit.text().strip()
        port = self.port_edit.text().strip()
        database = self.db_edit.text().strip()
        user = self.user_edit.text().strip()
        password = self.pass_edit.text().strip()

        if not (host and port and database and user):
            QMessageBox.warning(self, "Connection", "Введите параметры подключения (host, port, database, user).")
            return

        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        try:
            self.engine = create_engine(url, client_encoding='utf8', future=True)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.inspector = inspect(self.engine)
            self.tables = sorted(self.inspector.get_table_names())
            self.tables_list.clear()
            for t in self.tables:
                self.tables_list.addItem(QListWidgetItem(t))
            self.log(f"Connected to {database} @ {host}:{port}. Tables: {len(self.tables)}")
        except SQLAlchemyError as e:
            tb = traceback.format_exc()
            self.log("Connection failed:", str(e))
            QMessageBox.critical(self, "Connection error", f"Не удалось подключиться:\n{e}\n\n{tb}")

    def on_table_selected(self):
        sel = self.tables_list.selectedItems()
        self.cols_list.clear()
        if not sel:
            return
        table = sel[0].text()
        try:
            cols = [c["name"] for c in self.inspector.get_columns(table)]
            self.table_columns[table] = cols
            for c in cols:
                item = QListWidgetItem(c)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.cols_list.addItem(item)
        except Exception as e:
            self.log("Error getting columns for", table, ":", e)

    def add_selected_columns(self):
        sel_table_items = self.tables_list.selectedItems()
        if not sel_table_items:
            QMessageBox.warning(self, "No table", "Сначала выберите таблицу слева.")
            return
        table = sel_table_items[0].text()
        chosen = []
        for i in range(self.cols_list.count()):
            item = self.cols_list.item(i)
            if item.checkState() == Qt.Checked:
                chosen.append(item.text())

        if not chosen:
            QMessageBox.information(self, "No columns", "Не выбрано ни одной колонки.")
            return

        s = self.selected_cols.setdefault(table, set())
        s.update(chosen)
        self._refresh_features_text()
        self.log(f"Added {len(chosen)} columns from {table}")

    def _refresh_features_text(self):
        lines = []
        for t, cols in self.selected_cols.items():
            lines.append(f"{t}: {', '.join(sorted(cols))}")
        self.features_text.setPlainText("\n".join(lines))

    def on_build_dataframe(self):
        if not self.engine:
            QMessageBox.warning(self, "No connection", "Сначала подключитесь к БД.")
            return
        if not self.selected_cols:
            QMessageBox.warning(self, "No features", "Выберите хотя бы одну колонку.")
            return

        try:
            df = self.build_dataframe_from_selected(auto_join=self.autojoin_check.isChecked())
            self.current_df = df
            self.table_model.setDataFrame(df)
            self.save_csv_btn.setEnabled(True)
            self.hist_btn.setEnabled(True)
            self.multi_btn.setEnabled(True)

            self.log("DataFrame built:", df.shape)
        except Exception as e:
            tb = traceback.format_exc()
            self.log("Error building DataFrame:", e)
            QMessageBox.critical(self, "Build error", f"Ошибка при сборке DataFrame:\n{e}\n\n{tb}")

    def build_dataframe_from_selected(self, auto_join=True) -> pd.DataFrame:
        tables = list(self.selected_cols.keys())
        if not tables:
            raise ValueError("No tables selected")

        all_needed = set(tables)
        if auto_join:
            for (l_t, l_c, r_t, r_c) in self.join_map:
                if l_t in all_needed and r_t not in all_needed:
                    all_needed.add(r_t)
                if r_t in all_needed and l_t not in all_needed:
                    all_needed.add(l_t)
        tables_union = list(all_needed)

        start = None
        preferred = ["matches", "teams", "players", "stadiums", tables_union[0]]
        for p in preferred:
            if p in tables_union:
                start = p
                break
        if start is None:
            start = tables_union[0]

        select_parts = []
        for t, cols in self.selected_cols.items():
            for c in sorted(cols):
                alias = f"{t}__{c}"
                select_parts.append(f'{t}."{c}" AS "{alias}"')

        if not select_parts:
            for t in tables_union:
                cols = [c["name"] for c in self.inspector.get_columns(t)]
                for c in cols:
                    alias = f"{t}__{c}"
                    select_parts.append(f'{t}."{c}" AS "{alias}"')

        from_clause = f'"{start}"'
        joins = []
        used = {start}

        if auto_join:
            progress = True
            while progress:
                progress = False
                for (l_t, l_c, r_t, r_c) in self.join_map:
                    if l_t in used and r_t not in used and r_t in tables_union:
                        joins.append((r_t, f'"{l_t}"."{l_c}" = "{r_t}"."{r_c}"'))
                        used.add(r_t)
                        progress = True
                    elif r_t in used and l_t not in used and l_t in tables_union:
                        joins.append((l_t, f'"{r_t}"."{r_c}" = "{l_t}"."{l_c}"'))
                        used.add(l_t)
                        progress = True

            for t in tables_union:
                if t not in used:
                    joins.append((t, "1=1"))
                    used.add(t)
        else:
            for t in tables_union:
                if t != start:
                    joins.append((t, "1=1"))

        select_sql = ",\n    ".join(select_parts)
        sql = f"SELECT\n    {select_sql}\nFROM {from_clause}\n"
        for (t, cond) in joins:
            sql += f'LEFT JOIN "{t}" ON {cond}\n'

        self.log("Executing SQL (preview):")
        self.log(sql[:1000] + ("..." if len(sql) > 1000 else ""))

        df = pd.read_sql(sql, self.engine)
        return df

    def on_histograms(self):
        if self.current_df is None or self.current_df.empty:
            QMessageBox.information(self, "No data", "Сначала соберите DataFrame (Build DataFrame).")
            return
        dlg = HistogramDialog(self, self.current_df)
        dlg.exec()

    def on_save_csv(self):
        df = self.current_df
        if df is None or df.empty:
            QMessageBox.information(self, "No data", "Нет данных для сохранения.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv);;All files (*)")
        if not fname:
            return
        df.to_csv(fname, index=False)
        QMessageBox.information(self, "Saved", f"Saved to {fname}")

    def on_clear_features(self):
        reply = QMessageBox.question(
            self,
            "Clear features",
            "Очистить все выбранные признаки и текущий DataFrame?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.selected_cols.clear()
        self._refresh_features_text()

        self.current_df = pd.DataFrame()
        self.table_model.setDataFrame(self.current_df)

        self.hist_btn.setEnabled(False)
        self.multi_btn.setEnabled(False)
        self.save_csv_btn.setEnabled(False)

        for i in range(self.cols_list.count()):
            item = self.cols_list.item(i)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)

        self.log("Selected features cleared")



def main():
    app = QApplication(sys.argv)
    w = DBFeatureSelector()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
