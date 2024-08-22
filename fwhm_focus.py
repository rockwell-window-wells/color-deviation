import cv2
import numpy as np
from picamera2 import Picamera2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from time import sleep

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def calculate_fwhm(image):
    # Select the region of interest (ROI) for focus analysis
    roi = image[:, :]  # Adjust the coordinates as needed

    # Convert the image to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Sum the pixel intensities along one axis to create a 1D profile
    profile = np.mean(gray, axis=0)

    # Find the peak and fit a Gaussian curve to the profile
    peak_idx = np.argmax(profile)
    p0 = [profile[peak_idx], peak_idx, 10]  # Initial guess for Gaussian parameters
    try:
        popt, _ = curve_fit(gaussian, np.arange(len(profile)), profile, p0=p0)
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[2])  # FWHM calculation
    except RuntimeError:
        fwhm = np.inf  # If fitting fails, return an infinite FWHM

    return fwhm

def main():
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (4056, 3040)},
    )
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            # Capture a frame from the camera
            frame = picam2.capture_array()

            # Calculate the FWHM of the frame
            fwhm = calculate_fwhm(frame)

            # Display the FWHM value in the terminal
            print(f"FWHM: {fwhm:.2f}")
            # Display the live preview (optional)
            # cv2.imshow("Preview", frame)

            # Break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
              #  break
            sleep(1)
    finally:
        picam2.stop()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()