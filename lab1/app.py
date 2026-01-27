import sys
import traceback

from PySide6.QtWidgets import QApplication

from db_feature_selector import DBFeatureSelector



def main():
    app = QApplication(sys.argv)
    w = DBFeatureSelector()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
