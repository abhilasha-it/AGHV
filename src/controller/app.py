"""Entry point for the controller/simulator desktop app.

Usage:
    python -m src.controller.app
"""

import sys

from PyQt6.QtWidgets import QApplication

from src.controller.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
