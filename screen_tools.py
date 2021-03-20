import ctypes
import win32gui
import win32ui
import win32con
import numpy
from PIL import Image

# this actually only needs to be runs once per session, but fixes different
# monitors screen grabbing the resolution. Solution from here;
# https://stackoverflow.com/questions/44398075/can-dpi-scaling-be-enabled-disabled-programmatically-on-a-per-session-basis
PROCESS_PER_MONITOR_DPI_AWARE = 2
user32 = ctypes.windll.user32
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
user32.SetProcessDPIAware()


class Screen(object):

    def grab_screen(self, x1=0, y1=0, x2=0, y2=0):
        """
        Grab an area of the screen and return as numpy array
        """
        hwin = win32gui.GetDesktopWindow()

        width = x2 - x1 + 1
        height = y2 - y1 + 1

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()

        # create a blank bitmap image and bind to DC
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)

        # copy the screen into buffer
        memdc.BitBlt((0, 0), (width, height), srcdc, (x1, y1), win32con.SRCCOPY)
        signedIntsArray = bmp.GetBitmapBits(True)

        # convert to numpy array
        img = numpy.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        # cleanup
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        # NOTE: this image is BGRA
        return img

    def save_img(self, img, path):
        img = Image.fromarray(img)
        img.save(path)
