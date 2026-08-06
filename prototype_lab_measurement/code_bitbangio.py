# SPDX-License-Identifier: MIT
"""
TCS3430 command/response sensor node for Raspberry Pi Pico W — bitbangio variant.

Same as code.py, but uses software (bit-banged) I2C instead of the RP2040's
hardware I2C peripheral, to test whether the hang is specific to the
hardware I2C block's handling of clock stretching.
"""

import time
import board
import bitbangio

from adafruit_tcs3430 import TCS3430, ALSGain, InterruptPersistence

# SCL = physical pin 2 (GP1), SDA = physical pin 1 (GP0) — the Pico's default I2C0 bus
i2c = bitbangio.I2C(board.GP1, board.GP0, frequency=50_000)
tcs = TCS3430(i2c)

tcs.als_gain = ALSGain.GAIN_64X
tcs.integration_time = 100.0

# Enable the ALS "data ready" interrupt flag and wait on it before each
# read, matching Adafruit's reference example. Reading tcs.channels before
# the first integration cycle completes (right after a fresh boot) appears
# to be what was causing reads to hang indefinitely.
tcs.als_interrupt_enabled = True
tcs.interrupt_persistence = InterruptPersistence.EVERY
tcs.clear_als_interrupt()

SAMPLES_PER_READING = 5
DATA_READY_TIMEOUT = 2.0  # seconds — never wait on the sensor forever


def read_one_sample():
    """Wait for a fresh reading and return (x, y, z, ir1). Raises
    RuntimeError if the sensor doesn't report data-ready in time, instead
    of blocking forever."""
    start = time.monotonic()
    while not tcs.als_interrupt:
        if time.monotonic() - start > DATA_READY_TIMEOUT:
            raise RuntimeError("sensor data-ready timeout")
        time.sleep(0.01)
    x, y, z, ir1 = tcs.channels
    tcs.clear_als_interrupt()
    return x, y, z, ir1


print("READY")

while True:
    try:
        line = input().strip()

        if line == "READ":
            sx = sy = sz = si = 0.0
            for _ in range(SAMPLES_PER_READING):
                x, y, z, ir1 = read_one_sample()
                sx += x
                sy += y
                sz += z
                si += ir1

            n = SAMPLES_PER_READING
            print(f"{sx / n:.2f},{sy / n:.2f},{sz / n:.2f},{si / n:.2f}")
        elif line:
            print(f"ERR unknown command: {line}")

    except Exception as e:
        # A dropped/reconnecting serial connection, or a sensor data-ready
        # timeout, can throw here. Swallow it and keep the loop alive rather
        # than crashing out of listening for commands.
        print(f"ERR {type(e).__name__}: {e}")
        time.sleep(0.2)
