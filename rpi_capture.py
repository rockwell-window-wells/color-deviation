import warnings
from picamera2 import Picamera2
from picamera2.previews import QtPreview
import time

warnings.filterwarnings("ignore", category=UserWarning, module='libcamera')

picam2 = Picamera2()

picam2.start_preview(QtPreview())

picam2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

picam2.stop()
picam2.stop_preview()

print("Preview stopped.")