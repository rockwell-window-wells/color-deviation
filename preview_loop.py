from picamera2 import Picamera2, Preview

# Initialize the Picamera2
picam2 = Picamera2()

# Set the preview configuration
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Start the preview using the built-in preview option
picam2.start_preview(Preview.QTGL)

# Start the camera
picam2.start()

# Keep the preview running
input("Press Enter to exit...")

# Stop the camera and preview
picam2.stop()