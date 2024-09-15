import sys

from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QTabWidget
from PySide6.QtGui import QScreen, QPixmap, QImage
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtMultimedia import QWindowCapture, QMediaCaptureSession, QVideoFrame, QVideoFrameFormat
from PySide6.QtMultimediaWidgets import QVideoWidget
import time
import ctypes

import numpy
import cv2
import mouse

from wizard_eyes.client import Client


class ScreenCapture(QThread):

    new_image = Signal(QPixmap)

    def __init__(self, parent):
        super().__init__(parent)
        self.screen = QScreen()

    def run(self):
        while True:
            g = self.screen.geometry()
            pixmap = self.screen.grabWindow(
                0, g.x(), g.y(), g.width(), g.height()
            )
            time.sleep(0.05)


    def get_screen_image(self):
        screen = QScreen()
        image = screen.grabWindow(0).toImage()
        return image


class Visualiser(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Wizard Eyes Visualiser')
        self.setGeometry(100, 100, 800, 600)
        self.label = QLabel(self)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.time = time.time()
        self.client = Client('RuneLite')

        self.session = QMediaCaptureSession()
        self.capture = QWindowCapture()
        for window in self.capture.capturableWindows():
            if "RuneLite" in window.description():
                self.capture.setWindow(window)
                break

        self.session.setWindowCapture(self.capture)
        self.video = QVideoWidget()
        self.session.setVideoOutput(self.video)

        tabs = QTabWidget()
        self.layout.addWidget(tabs)
        tabs.addTab(self.video, 'Original')

        self.video2 = QVideoWidget()
        tabs.addTab(self.video2, 'Processed')

        self.video.videoSink().videoFrameChanged.connect(self.update_image)

        self.capture.start()

    def update_image(self, frame: QVideoFrame):
        time_now = time.time()
        time_since = round(time_now - self.time, 3)
        self.time = time_now
        image = frame.toImage()

        image = image.convertToFormat(QImage.Format.Format_RGB32)

        ptr = image.constBits()
        arr = numpy.array(ptr).reshape((image.height(), image.width(), 4))
        # arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

        # patch mouse
        x, y = mouse.get_position()

        img_processed = self.client.process_img(arr)
        self.client._hsv_img = self.client.convert_to_hsv(arr)
        self.client._original_img = arr
        self.client._img = img_processed
        self.client.update()
        self.client.tabs.update()
        for draw in self.client.draw_calls:
            try:
                draw()
            except Exception as e:
                self.client.logger.debug(f'Error in draw call: {e}')

        # convert updated image to QVideoFrame
        arr = self.client.original_img

        # cv2.imshow('test', arr)
        # cv2.moveWindow('test', 5, 20)
        # cv2.waitKey(1)

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
        image = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_RGBA8888)

        # QVideoFrame frame(QVideoFrameFormat(img.size(), QVideoFrameFormat::pixelFormatFromImageFormat(img.format()));
        # frame.map(QVideoFrame::ReadWrite);
        # memcpy(frame.bits(0), img.bits(), img.sizeInBytes());
        # frame.unmap();
        # m_videoSink->setVideoFrame(frame);

        format_ = QVideoFrameFormat(
            image.size(),
            QVideoFrameFormat.pixelFormatFromImageFormat(image.format())
        )
        frame2 = QVideoFrame(format_)
        frame2.map(QVideoFrame.ReadWrite)

        end = len(image.bits().tobytes())
        frame2.bits(0)[:end] = image.bits()

        # use ctypes to copy the data
        # ctypes.memmove(
        #     frame2.bits(0)[0],
        #     # ctypes.cast(id(frame2.bits(0)), ctypes.POINTER(ctypes.c_ubyte)),
        #     image.bits()[0],
        #     image.sizeInBytes()
        # )

        frame2.unmap()

        self.video2.videoSink().setVideoFrame(frame2)

        print(time_since, arr.shape)



def main():
    app = QApplication([])
    v = Visualiser(None)
    v.show()
    # screen = QScreen()
    app.exec()


if __name__ == '__main__':
    main()
