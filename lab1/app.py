# app.py
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QListWidget,
    QHBoxLayout, QMessageBox, QTableWidget, QTableWidgetItem, QFileDialog
)
from PySide6.QtCore import Qt
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5433/football_db")
FALLBACK_SQLITE = "sqlite:///football.db"

def get_engine():
    try:
        engine = create_engine(DATABASE_URL)
        conn = engine.connect()
        conn.close()
        return engine
    except Exception:
        return create_engine(FALLBACK_SQLITE)

def load_joined_df(engine):
    # build LEFT JOIN query - keep to safe columns
    with engine.connect() as conn:
        # join players + teams + matches + stadiums + goals (aggregate goals count maybe)
        q = text("""
        SELECT
          p.PLAYER_ID, p.FIRST_NAME, p.LAST_NAME, p.NATIONALITY as PLAYER_NATION, p.DOB,
          p.TEAM as PLAYER_TEAM,
          t.TEAM_NAME, t.COUNTRY as TEAM_COUNTRY, t.HOME_STADIUM,
          m.MATCH_ID, m.SEASON, m.DATE_TIME, m.HOME_TEAM, m.AWAY_TEAM, m.STADIUM as MATCH_STADIUM,
          m.HOME_TEAM_SCORE, m.AWAY_TEAM_SCORE, m.ATTENDANCE,
          s.NAME as ST_NAME, s.COUNTRY as ST_COUNTRY, s.CAPACITY
        FROM players p
        LEFT JOIN teams t ON p.TEAM = t.TEAM_NAME
        LEFT JOIN matches m ON (m.HOME_TEAM = t.TEAM_NAME OR m.AWAY_TEAM = t.TEAM_NAME)
        LEFT JOIN stadiums s ON s.NAME = m.STADIUM
        LIMIT 10000
        """)
        df = pd.read_sql(q, conn)
    return df

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Football Data Explorer (Well-done)")
        self.resize(1100, 700)
        self.engine = get_engine()
        self.df = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.status_label = QLabel("DB: " + str(self.engine.url))
        top.addWidget(self.status_label)
        btn_load = QPushButton("Load joined table")
        btn_load.clicked.connect(self.load_df)
        top.addWidget(btn_load)
        layout.addLayout(top)

        # preview table
        self.table = QTableWidget()
        layout.addWidget(self.table, 3)

        # selectors
        sel_layout = QHBoxLayout()
        self.num_list = QListWidget()
        self.num_list.setSelectionMode(QListWidget.MultiSelection)
        sel_layout.addWidget(self.num_list)
        btn_hist = QPushButton("Plot 2 Histograms (select 2)")
        btn_hist.clicked.connect(self.plot_histograms)
        sel_layout.addWidget(btn_hist)

        self.mult_list = QListWidget()
        self.mult_list.setSelectionMode(QListWidget.MultiSelection)
        sel_layout.addWidget(self.mult_list)
        btn_multi = QPushButton("Plot Multivariate (select 3-4)")
        btn_multi.clicked.connect(self.plot_multivariate)
        sel_layout.addWidget(btn_multi)

        layout.addLayout(sel_layout)

        self.setLayout(layout)

    def load_df(self):
        try:
            df = load_joined_df(self.engine)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            return
        self.df = df
        self.populate_preview()
        # numeric columns
        numeric = list(self.df.select_dtypes(include=[np.number]).columns)
        self.num_list.clear()
        for c in numeric:
            self.num_list.addItem(c)
        # all usable columns for multivariate (numbers + some encoded)
        allcols = list(self.df.columns)
        self.mult_list.clear()
        for c in allcols:
            self.mult_list.addItem(c)

    def populate_preview(self):
        df = self.df.head(200)
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                v = row[col]
                it = QTableWidgetItem(str(v))
                self.table.setItem(i, j, it)

    def plot_histograms(self):
        sel = [it.text() for it in self.num_list.selectedItems()]
        if len(sel) != 2:
            QMessageBox.warning(self, "Pick 2", "Select exactly 2 numeric columns for histograms.")
            return
        a, b = sel
        fig, axes = plt.subplots(1, 2)
        self.df[a].dropna().hist(ax=axes[0])
        axes[0].set_title(a)
        self.df[b].dropna().hist(ax=axes[1])
        axes[1].set_title(b)
        plt.tight_layout()
        plt.show()

    def plot_multivariate(self):
        sel = [it.text() for it in self.mult_list.selectedItems()]
        if len(sel) < 3 or len(sel) > 4:
            QMessageBox.warning(self, "Pick 3-4", "Select 3 or 4 columns for multivariate plot.")
            return
        chosen = sel
        df2 = self.df[chosen].copy()
        # try numeric conversion where possible
        for c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
        # if 3 columns -> 3D scatter if available, otherwise pairplot
        if df2.shape[1] == 3:
            try:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                xs = df2.iloc[:,0].values
                ys = df2.iloc[:,1].values
                zs = df2.iloc[:,2].values
                ax.scatter(xs, ys, zs)
                ax.set_xlabel(df2.columns[0])
                ax.set_ylabel(df2.columns[1])
                ax.set_zlabel(df2.columns[2])
                plt.show()
                return
            except Exception:
                pass
        # else do scatter matrix
        scatter_matrix(df2.dropna(), diagonal='hist')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
