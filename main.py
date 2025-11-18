"""
Main application launch file
"""
import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def main():
    """Launches the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("Acquisition System")
    app.setOrganizationName("Computer Vision Lab")
    
    # Set style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
