"""
Serial link to the Pico W TCS3430 sensor node.

Runs on any host with pyserial installed (Windows now, Jetson Nano later).
"""

import time
import serial
from serial.tools import list_ports

# USB vendor IDs to match against. The Pico/Pico W shows up under the
# Raspberry Pi Foundation's VID only in bootloader (RPI-RP2) mode — once
# CircuitPython is running, the USB CDC serial port identifies under
# Adafruit's VID instead, since that's whose CircuitPython build is on it.
KNOWN_VIDS = {0x2E8A, 0x239A}


def find_pico_port():
    """Return the device path (e.g. 'COM6') of a connected Pico, or None."""
    for port in list_ports.comports():
        if port.vid in KNOWN_VIDS:
            return port.device
        if port.description and "CircuitPython" in port.description:
            return port.device
    return None


class TCS3430Link:
    def __init__(self, port, baudrate=115200, timeout=3, boot_timeout=6):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self._soft_reboot()
        self._wait_for_boot(boot_timeout)

    def _soft_reboot(self):
        """Interrupt whatever's running (Ctrl-C) and trigger a soft reload
        (Ctrl-D), the same sequence tools like Thonny send when they connect.
        This re-runs code.py from scratch, including recreating the I2C bus
        object — which clears a stuck/wedged I2C state left over from the
        previous session. Without this, the board can inherit a hung bus
        that a passive connection alone won't recover from."""
        self.ser.write(b"\x03")  # Ctrl-C: interrupt running code.py
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        self.ser.write(b"\x04")  # Ctrl-D: soft reload
        time.sleep(0.3)

    def _wait_for_boot(self, boot_timeout):
        """Wait for the board's READY line after the reset that opening the
        serial port triggers. If READY never shows up (e.g. the board was
        already running past that point), just proceed after the timeout —
        it's not necessarily an error."""
        deadline = time.time() + boot_timeout
        while time.time() < deadline:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "READY":
                break
        self.ser.reset_input_buffer()

    def read_sample(self, max_attempts=5):
        """Send a READ command and return (x, y, z, ir1) as floats.

        CircuitPython's serial console echoes back whatever you send it
        (the "READ" command itself) before printing the real response, so
        we skip that echo line — and anything else that isn't a 4-value
        CSV line — while looking for the actual reading.
        """
        self.ser.write(b"READ\n")

        for _ in range(max_attempts):
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                raise TimeoutError("No response from sensor board")
            if line in ("READ", "READY"):
                continue  # echoed command, or the board's boot banner
            if line.startswith("ERR"):
                raise RuntimeError(f"Sensor board reported an error: {line}")

            parts = line.split(",")
            if len(parts) == 4:
                try:
                    x, y, z, ir1 = (float(p) for p in parts)
                    return x, y, z, ir1
                except ValueError:
                    continue  # not a valid reading line, keep looking

        raise ValueError("No valid reading from sensor board after retries")

    def close(self):
        self.ser.close()
