from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QWidget, QLabel

import numpy


class Visualiser(QWidget):

    new_image: Signal = Signal(numpy.ndarray)

    def __init__(self, parent: Optional[QWidget], client):
        super().__init__(parent)
        self._client = client
        self.setWindowTitle('Wizard Eyes Visualiser')
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.new_image.connect(self.set_image)

    def set_image(self, image: numpy.ndarray):
        """Set the image to display in the visualiser.

        :param image: The image to display.

        """
        h, w, c = image.shape
        pixmap = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
        self.label.setPixmap(pixmap)


def run(client):
    """Start the PySide visualiser application in a separate thread.

    :param client: The client object to use for the visualiser.

    """
    app = QApplication([])
    visualiser = Visualiser(None, client)
    visualiser.show()
    app.exec_()
