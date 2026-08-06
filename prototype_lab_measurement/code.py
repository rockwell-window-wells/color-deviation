# SPDX-License-Identifier: MIT
"""
TCS3430 command/response sensor node for Raspberry Pi Pico W.
Save this file as code.py on the CIRCUITPY drive.

Continuously polls the sensor in the background on a ~1s cadence — this
keeps the sensor from ever sitting idle, which is what was causing on-demand
reads to hang indefinitely.

Drives a 16-LED WS2812B ring (via the built-in neopixel library) as a fixed
illumination source, lit at boot and left on by default, to keep lighting
consistent between calibration and measurement instead of relying on
ambient room light. LIGHT_ON / LIGHT_OFF commands are also available.

When a "READ" command arrives over USB serial, it does NOT just grab
whatever's currently latched (which could reflect light captured before the
target was placed / before Enter was pressed). Instead it waits a full two
integration-time cycles first — enough to guarantee the cycle already in
progress at command-receipt time (which may have started earlier) is
discarded, and the next one it reads started entirely after the command.
It then takes SAMPLES_PER_READ such readings (spaced one integration time
apart) and replies with their average: X,Y,Z,IR1

A hardware watchdog resets the board automatically if the main loop ever
stalls (e.g. a genuine I2C-level hang) for longer than WATCHDOG_TIMEOUT
seconds — a full reset is what has reliably cleared past hangs, so this
makes that recovery automatic instead of requiring a manual Ctrl-D.

This board does no calibration or color-space math itself — it's a dumb
sensor peripheral. All calibration and Lab conversion happens on the
host computer (and later, the Jetson Nano).
"""

import sys
import time
import board
import busio
import supervisor
import microcontroller
import neopixel
from watchdog import WatchDogMode

from adafruit_tcs3430 import TCS3430, ALSGain

# Default I2C0 bus: SCL = physical pin 2 (GP1), SDA = physical pin 1 (GP0)
# (board.I2C() doesn't exist on this board's module — use busio explicitly)
i2c = busio.I2C(board.GP1, board.GP0)
tcs = TCS3430(i2c)

tcs.als_gain = ALSGain.GAIN_64X

INTEGRATION_TIME_MS = 200.0  # longer for better per-sample SNR — tune freely
tcs.integration_time = INTEGRATION_TIME_MS
INTEGRATION_TIME_S = INTEGRATION_TIME_MS / 1000.0

SAMPLES_PER_READ = 4  # independent samples averaged together per READ
SETTLE_MARGIN = 0.01  # small safety margin added to every wait, in seconds

POLL_INTERVAL = 1.0  # background keep-alive cadence

# --- LED ring (illumination source) ---
# Data line = physical pin 4 (GP2). VCC -> physical pin 40 (VBUS, 5V from
# USB), GND -> any GND pin e.g. physical pin 38.
NUM_PIXELS = 16
LIGHT_BRIGHTNESS = 0.2  # keep well under USB's ~500mA budget at 16 LEDs
pixels = neopixel.NeoPixel(board.GP2, NUM_PIXELS, brightness=LIGHT_BRIGHTNESS, auto_write=False)


def light_on():
    pixels.fill((255, 255, 255))
    pixels.show()


def light_off():
    pixels.fill((0, 0, 0))
    pixels.show()


light_on()  # illuminated by default at boot

# Worst-case time through one full loop iteration handling a READ command,
# plus generous headroom. If the loop doesn't come back around to feed the
# watchdog within this many seconds, the board force-resets itself.
WATCHDOG_TIMEOUT = max(8.0, (SAMPLES_PER_READ + 2) * INTEGRATION_TIME_S + 4.0)
microcontroller.watchdog.timeout = WATCHDOG_TIMEOUT
microcontroller.watchdog.mode = WatchDogMode.RESET

last_poll = time.monotonic() - POLL_INTERVAL  # poll immediately on boot

print("READY")

while True:
    microcontroller.watchdog.feed()
    now = time.monotonic()

    if now - last_poll >= POLL_INTERVAL:
        try:
            tcs.channels  # background "keep-alive" read; result not used
        except Exception as e:
            print(f"ERR poll {type(e).__name__}: {e}")
        last_poll = now

    if supervisor.runtime.serial_bytes_available:
        line = sys.stdin.readline().strip()
        if line == "READ":
            try:
                # Guarantee the first sample comes from a cycle that started
                # entirely after this command was received, not a cycle
                # already partway through (which could predate the target
                # being placed).
                time.sleep(2 * INTEGRATION_TIME_S + SETTLE_MARGIN)

                samples = [tcs.channels]
                for _ in range(SAMPLES_PER_READ - 1):
                    time.sleep(INTEGRATION_TIME_S + SETTLE_MARGIN)
                    samples.append(tcs.channels)

                n = len(samples)
                sx = sum(s[0] for s in samples)
                sy = sum(s[1] for s in samples)
                sz = sum(s[2] for s in samples)
                si = sum(s[3] for s in samples)
                print(f"{sx / n:.2f},{sy / n:.2f},{sz / n:.2f},{si / n:.2f}")
                last_poll = time.monotonic()  # this counts as the latest poll too
            except Exception as e:
                print(f"ERR read {type(e).__name__}: {e}")
        elif line == "LIGHT_ON":
            light_on()
            print("OK")
        elif line == "LIGHT_OFF":
            light_off()
            print("OK")
        elif line:
            print(f"ERR unknown command: {line}")

    time.sleep(0.01)
