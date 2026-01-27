

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex

import pandas as pd


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
