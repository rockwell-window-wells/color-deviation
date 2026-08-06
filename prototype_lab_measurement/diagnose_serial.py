"""
Raw diagnostic: mirrors sensor_link.py's connect sequence (soft reboot via
Ctrl-C/Ctrl-D, then wait for READY) and prints exactly what bytes come back
at each stage, with no line-parsing assumptions.
"""

import time
import serial
from sensor_link import find_pico_port

port = find_pico_port()
if not port:
    port = input("Port: ").strip()
print(f"Opening {port}...")

ser = serial.Serial(port, 115200, timeout=1)

print("\n--- Stage 1: sending Ctrl-C (interrupt) ---")
ser.write(b"\x03")
time.sleep(0.5)
buf = ser.read(ser.in_waiting or 1)
print(f"Received {len(buf)} bytes: {buf!r}")

print("\n--- Stage 2: clearing buffer, sending Ctrl-D (soft reload) ---")
ser.reset_input_buffer()
ser.write(b"\x04")
time.sleep(0.5)

print("\n--- Stage 3: listening 6 seconds for boot output / READY ---")
end = time.time() + 6
buf = b""
while time.time() < end:
    chunk = ser.read(ser.in_waiting or 1)
    if chunk:
        buf += chunk
print(f"Received {len(buf)} bytes: {buf!r}")

print("\n--- Stage 4: sending READ ---")
ser.write(b"READ\n")

print("\n--- Stage 5: listening 8 seconds for response ---")
end = time.time() + 8
buf = b""
while time.time() < end:
    chunk = ser.read(ser.in_waiting or 1)
    if chunk:
        buf += chunk
print(f"Received {len(buf)} bytes: {buf!r}")

ser.close()

