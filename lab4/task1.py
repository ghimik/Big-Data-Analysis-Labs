import sys
import os
import math
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTextEdit, QLabel, QListWidget, QListWidgetItem,
    QTabWidget, QGroupBox, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PandasModel:
    """Simple model for showing small DataFrame in QTextEdit as text."""
    @staticmethod
    def to_text(df: pd.DataFrame, max_rows=200):
        if df is None:
            return "No data"
        with pd.option_context('display.max_rows', max_rows, 'display.max_columns', 20):
            # Принудительно выравниваем все столбцы по ширине
            return df.to_string(col_space=10, justify='right')



class WorkerTrain(QThread):
    finished = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, pipeline, X_train, X_test, y_train, y_test):
        super().__init__()
        self.pipeline = pipeline
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        try:
            self.pipeline.fit(self.X_train, self.y_train)
            y_pred = self.pipeline.predict(self.X_test)
            y_train_pred = self.pipeline.predict(self.X_train)
            self.finished.emit(self.pipeline, (y_pred, y_train_pred), None)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(tb)


class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Desktop — Regression helper")
        self.resize(1500, 700)

        self.df = None
        self.loaded_path = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        top_row.addWidget(self.load_btn)
        self.path_label = QLabel("No file loaded")
        top_row.addWidget(self.path_label)
        top_row.addStretch()
        v.addLayout(top_row)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs)

        self.tab_eda = QWidget()
        self.tabs.addTab(self.tab_eda, "EDA")
        self._build_eda_tab()

        self.tab_prep = QWidget()
        self.tabs.addTab(self.tab_prep, "Preprocessing")
        self._build_prep_tab()

        self.tab_model = QWidget()
        self.tabs.addTab(self.tab_model, "Modeling")
        self._build_model_tab()

    def _build_eda_tab(self):
        layout = QVBoxLayout(self.tab_eda)

        info_box = QGroupBox("Dataset information")
        info_layout = QFormLayout()
        info_box.setLayout(info_layout)
        self.shape_label = QLabel("rows: -, cols: -")
        self.memory_label = QLabel("memory: - bytes")
        info_layout.addRow("Shape:", self.shape_label)
        info_layout.addRow("Memory:", self.memory_label)

        layout.addWidget(info_box)

        stats_box = QGroupBox("Statistics")
        stats_layout = QHBoxLayout()
        stats_box.setLayout(stats_layout)

        self.num_stats_text = QTextEdit()
        self.num_stats_text.setReadOnly(True)
        self.num_stats_text.setFontFamily("Courier New")  
        stats_layout.addWidget(self.num_stats_text)

        self.cat_stats_text = QTextEdit()
        self.cat_stats_text.setReadOnly(True)
        self.cat_stats_text.setFontFamily("Courier New")  
        stats_layout.addWidget(self.cat_stats_text)


        layout.addWidget(stats_box)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh EDA")
        self.btn_refresh.clicked.connect(self.refresh_eda)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def refresh_eda(self):
        if self.df is None:
            QMessageBox.information(self, "No data", "Load a CSV first")
            return
        df = self.df
        self.shape_label.setText(f"rows: {df.shape[0]}, cols: {df.shape[1]}")
        mem = df.memory_usage(deep=True).sum()
        self.memory_label.setText(f"{mem} bytes")

        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] > 0:
            stats = []
            for c in numeric.columns:
                s = numeric[c].dropna()
                stats.append({
                    'column': c,
                    'min': s.min(),
                    '25%': s.quantile(0.25) if not s.empty else None,
                    'median': s.median() if not s.empty else None,
                    'mean': s.mean() if not s.empty else None,
                    '75%': s.quantile(0.75) if not s.empty else None,
                    'max': s.max() if not s.empty else None,
                    'missing': int(s.size - s.count())
                })
            stat_df = pd.DataFrame(stats).set_index('column')
            self.num_stats_text.setPlainText(PandasModel.to_text(stat_df))
        else:
            self.num_stats_text.setPlainText('No numeric columns')

        cat = df.select_dtypes(include=['object', 'category', 'bool'])
        if cat.shape[1] > 0:
            cats = []
            for c in cat.columns:
                s = cat[c]
                mode = None
                mode_count = 0
                if not s.dropna().empty:
                    mode = s.mode().iloc[0]
                    mode_count = int((s == mode).sum())
                cats.append({'column': c, 'mode': mode, 'mode_count': mode_count, 'unique': int(s.nunique(dropna=True)), 'missing': int(s.isna().sum())})
            cat_df = pd.DataFrame(cats).set_index('column')
            self.cat_stats_text.setPlainText(PandasModel.to_text(cat_df))
        else:
            self.cat_stats_text.setPlainText('No categorical columns')

    def _build_prep_tab(self):
        layout = QVBoxLayout(self.tab_prep)

        top = QHBoxLayout()
        self.cols_list = QListWidget()
        self.cols_list.setSelectionMode(QListWidget.MultiSelection)
        top.addWidget(self.cols_list)

        ops = QVBoxLayout()
        ops.addWidget(QLabel("Missing values:"))
        self.fill_median_btn = QPushButton("Fill selected numeric with median")
        self.fill_median_btn.clicked.connect(self.fill_median)
        ops.addWidget(self.fill_median_btn)
        self.fill_mode_btn = QPushButton("Fill selected categorical with mode")
        self.fill_mode_btn.clicked.connect(self.fill_mode)
        ops.addWidget(self.fill_mode_btn)
        self.dropna_btn = QPushButton("Drop rows with any NA")
        self.dropna_btn.clicked.connect(self.dropna)
        ops.addWidget(self.dropna_btn)

        ops.addSpacing(10)
        ops.addWidget(QLabel("Outliers (z-score):"))
        row = QHBoxLayout()
        self.z_thresh_spin = QDoubleSpinBox(); self.z_thresh_spin.setRange(0.5, 10.0); self.z_thresh_spin.setSingleStep(0.1); self.z_thresh_spin.setValue(3.0)
        row.addWidget(QLabel("z thresh:")); row.addWidget(self.z_thresh_spin)
        ops.addLayout(row)
        self.remove_outliers_btn = QPushButton("Remove outliers (selected numeric)")
        self.remove_outliers_btn.clicked.connect(self.remove_outliers)
        ops.addWidget(self.remove_outliers_btn)

        ops.addSpacing(10)
        ops.addWidget(QLabel("Categorical encoding:"))
        self.onehot_btn = QPushButton("One-hot encode selected")
        self.onehot_btn.clicked.connect(self.onehot_encode)
        ops.addWidget(self.onehot_btn)

        ops.addStretch()
        top.addLayout(ops)
        layout.addLayout(top)

        bottom = QHBoxLayout()

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFontFamily("Courier New")
        bottom.addWidget(self.preview_text)

        self.apply_btn = QPushButton("Refresh preview")
        self.apply_btn.clicked.connect(self.refresh_preview)
        bottom.addWidget(self.apply_btn)
        layout.addLayout(bottom)


    def _populate_columns_list(self):
        self.cols_list.clear()
        if self.df is None:
            return
        for c in self.df.columns:
            item = QListWidgetItem(c)
            item.setCheckState(Qt.Unchecked)
            self.cols_list.addItem(item)

    def _selected_columns(self):
        items = [self.cols_list.item(i) for i in range(self.cols_list.count())]
        return [it.text() for it in items if it.checkState() == Qt.Checked]

    def fill_median(self):
        cols = self._selected_columns()
        if not cols:
            QMessageBox.information(self, 'Select', 'Select columns in list to fill')
            return
        for c in cols:
            if pd.api.types.is_numeric_dtype(self.df[c]):
                med = self.df[c].median()
                self.df[c].fillna(med, inplace=True)
        QMessageBox.information(self, 'Done', 'Filled median for selected numeric columns')
        self.refresh_eda()
        self.refresh_preview()
        self._populate_columns_list()
        self.populate_model_controls()


    def fill_mode(self):
        cols = self._selected_columns()
        if not cols:
            QMessageBox.information(self, 'Select', 'Select columns in list to fill')
            return
        for c in cols:
            mode = self.df[c].mode()
            if not mode.empty:
                self.df[c].fillna(mode.iloc[0], inplace=True)
        QMessageBox.information(self, 'Done', 'Filled mode for selected columns')
        self.refresh_eda()
        self.refresh_preview()
        self._populate_columns_list()
        self.populate_model_controls()


    def dropna(self):
        prev = len(self.df)
        self.df.dropna(inplace=True)
        QMessageBox.information(self, 'Done', f'Dropped rows with NA: {prev - len(self.df)} removed')
        self.refresh_eda()
        self.refresh_preview()
        self._populate_columns_list()
        self.populate_model_controls()


    def remove_outliers(self):
        cols = self._selected_columns()
        if not cols:
            QMessageBox.information(self, 'Select', 'Select numeric columns to apply outlier removal')
            return
        thresh = float(self.z_thresh_spin.value())
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
        if not numeric_cols:
            QMessageBox.information(self, 'No numeric', 'No numeric columns selected')
            return
        from scipy import stats
        mask = pd.Series([True] * len(self.df))
        for c in numeric_cols:
            col = self.df[c]
            z = pd.Series(np.abs((col - col.mean()) / (col.std(ddof=0) if col.std(ddof=0) else 1)))
            mask &= (z <= thresh) | (col.isna())
        prev = len(self.df)
        self.df = self.df[mask].reset_index(drop=True)
        QMessageBox.information(self, 'Done', f'Removed outliers: {prev - len(self.df)} rows')
        self.refresh_eda()
        self.refresh_preview()
        self._populate_columns_list()
        self.populate_model_controls()


    def onehot_encode(self):
        cols = self._selected_columns()
        if not cols:
            QMessageBox.information(self, 'Select', 'Select categorical columns to encode')
            return
        try:
            before = set(self.df.columns)
            self.df = pd.get_dummies(self.df, columns=cols, drop_first=False)
            after = set(self.df.columns)
            added = len(after - before)
            QMessageBox.information(self, 'Done', f'One-hot encoded {len(cols)} cols, added {added} columns')
            self.refresh_eda()
            self.refresh_preview()
            self._populate_columns_list()
            self.populate_model_controls()
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


    def refresh_preview(self):
        if self.df is None:
            self.preview_text.setPlainText('No data')
            return
        self.preview_text.setPlainText(PandasModel.to_text(self.df.head(50)))

    def _build_model_tab(self):
        layout = QVBoxLayout(self.tab_model)

        top = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        left.addWidget(QLabel('Target column:'))
        self.target_cb = QComboBox(); left.addWidget(self.target_cb)
        left.addWidget(QLabel('Feature columns (multi-select in list):'))
        self.features_list = QListWidget(); self.features_list.setSelectionMode(QListWidget.MultiSelection)
        left.addWidget(self.features_list)

        left.addWidget(QLabel('Train/test split (test size %):'))
        self.test_size_spin = QSpinBox(); self.test_size_spin.setRange(5,50); self.test_size_spin.setValue(20); left.addWidget(self.test_size_spin)

        left.addWidget(QLabel('Random seed:'))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0,9999); self.seed_spin.setValue(42); left.addWidget(self.seed_spin)

        left.addWidget(QLabel('Algorithm:'))
        self.algo_cb = QComboBox(); self.algo_cb.addItems(['KNN','Linear','Lasso','ElasticNet']); left.addWidget(self.algo_cb)

        param_box = QGroupBox('Hyperparameters (simple)')
        pform = QFormLayout(); param_box.setLayout(pform)
        self.k_spin = QSpinBox(); self.k_spin.setRange(1,50); self.k_spin.setValue(5)
        pform.addRow('n_neighbors (KNN):', self.k_spin)
        self.alpha_edit = QLineEdit('1.0')
        pform.addRow('alpha (Lasso/ElasticNet):', self.alpha_edit)
        self.l1_ratio_edit = QLineEdit('0.5')
        pform.addRow('l1_ratio (ElasticNet):', self.l1_ratio_edit)
        left.addWidget(param_box)

        self.train_btn = QPushButton('Train & Evaluate')
        self.train_btn.clicked.connect(self.train_and_evaluate)
        left.addWidget(self.train_btn)

        self.save_model_btn = QPushButton('Save model')
        self.save_model_btn.clicked.connect(self.save_model)
        left.addWidget(self.save_model_btn)

        right.addWidget(QLabel('Metrics / log:'))
        self.metrics_text = QTextEdit(); self.metrics_text.setReadOnly(True); right.addWidget(self.metrics_text)

        self.figure = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.figure)
        right.addWidget(self.canvas)

        top.addLayout(left, 2)
        top.addLayout(right, 3)
        layout.addLayout(top)


    def populate_model_controls(self):
        self.target_cb.clear(); self.features_list.clear()
        if self.df is None:
            return
        for c in self.df.columns:
            self.target_cb.addItem(c)
            item = QListWidgetItem(c); item.setCheckState(Qt.Unchecked); self.features_list.addItem(item)

    def _get_selected_features(self):
        items = [self.features_list.item(i) for i in range(self.features_list.count())]
        return [it.text() for it in items if it.checkState() == Qt.Checked]

    def train_and_evaluate(self):
        if self.df is None:
            QMessageBox.information(self, 'No data', 'Load CSV first')
            return
        target = self.target_cb.currentText()
        if not target:
            QMessageBox.information(self, 'Select', 'Select target column')
            return
        features = self._get_selected_features()
        if not features:
            QMessageBox.information(self, 'Select', 'Select at least one feature')
            return
        try:
            test_size = float(self.test_size_spin.value())/100.0
            seed = int(self.seed_spin.value())
            X = self.df[features].copy()
            y = self.df[target].copy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

            numeric_feats = [c for c in features if pd.api.types.is_numeric_dtype(self.df[c])]
            categorical_feats = [c for c in features if not pd.api.types.is_numeric_dtype(self.df[c])]

            num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('num', num_pipe, numeric_feats), ('cat', cat_pipe, categorical_feats)])

            algo = self.algo_cb.currentText()
            model = None
            if algo == 'KNN':
                model = KNeighborsRegressor(n_neighbors=int(self.k_spin.value()))
            elif algo == 'Linear':
                model = LinearRegression()
            elif algo == 'Lasso':
                alpha = float(self.alpha_edit.text())
                model = Lasso(alpha=alpha, max_iter=5000)
            elif algo == 'ElasticNet':
                alpha = float(self.alpha_edit.text())
                l1_ratio = float(self.l1_ratio_edit.text())
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

            pipeline = Pipeline([('pre', preprocessor), ('model', model)])

            self.metrics_text.setPlainText('Training...')
            self.train_btn.setEnabled(False)
            self.worker = WorkerTrain(pipeline, X_train, X_test, y_train, y_test)
            self.worker.finished.connect(self._on_training_finished)
            self.worker.error.connect(self._on_training_error)
            self.worker.start()
            self._last_pipeline = pipeline
            self._last_X_test = X_test
            self._last_y_test = y_test
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            self.train_btn.setEnabled(True)

    def _on_training_error(self, tb):
        self.metrics_text.setPlainText('Error during training:\n' + tb)
        self.train_btn.setEnabled(True)

    def _on_training_finished(self, pipeline, preds_tuple, _):
        try:
            y_pred, y_train_pred = preds_tuple
            X_test = self._last_X_test
            y_test = self._last_y_test
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test==0, np.nan, y_test))) * 100
            r2 = r2_score(y_test, y_pred)
            text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAPE: {np.nan_to_num(mape):.3f}%\nR2: {r2:.4f}\n"
            self.metrics_text.setPlainText(text)

            self.figure.clear()
            ax = self.figure.add_subplot(121)
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Predicted vs Actual')

            ax2 = self.figure.add_subplot(122)
            res = y_test - y_pred
            ax2.hist(res, bins=30)
            ax2.set_title('Residuals')
            self.canvas.draw()

            self.train_btn.setEnabled(True)
            self._trained_pipeline = pipeline
            self._trained_metrics = text
        except Exception as e:
            self.metrics_text.setPlainText('Error processing results:\n' + str(e) + '\n' + traceback.format_exc())
            self.train_btn.setEnabled(True)

    def save_model(self):
        if not hasattr(self, '_trained_pipeline'):
            QMessageBox.information(self, 'No model', 'Train model first')
            return
        fname, _ = QFileDialog.getSaveFileName(self, 'Save model', '', 'Pickle files (*.pkl);;All files (*)')
        if not fname:
            return
        try:
            joblib.dump(self._trained_pipeline, fname)
            QMessageBox.information(self, 'Saved', f'Model saved to {fname}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CSV', os.getcwd(), 'CSV files (*.csv);;All files (*)')
        if not fname:
            return
        try:
            df = pd.read_csv(fname)
            df.columns = [c.strip() for c in df.columns]
            for c in df.select_dtypes(include=['object']).columns:
                df[c] = df[c].astype(str).str.strip()
                df[c].replace({'': np.nan}, inplace=True)
            self.df = df
            self.loaded_path = fname
            self.path_label.setText(fname)
            self.refresh_eda()
            self._populate_columns_list()
            self.populate_model_controls()
            self.refresh_preview()
        except Exception as e:
            QMessageBox.critical(self, 'Error loading CSV', str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MLApp()
    win.show()
    sys.exit(app.exec())
