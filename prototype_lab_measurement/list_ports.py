"""
Diagnostic: list every serial port Windows currently sees, along with its
USB vendor/product ID and description. Run this with the Pico plugged in
(and Thonny disconnected from it) to see what it shows up as.
"""

from serial.tools import list_ports

ports = list(list_ports.comports())

if not ports:
    print("No serial ports detected at all.")
else:
    for p in ports:
        vid = f"{p.vid:04X}" if p.vid is not None else "----"
        pid = f"{p.pid:04X}" if p.pid is not None else "----"
        print(f"{p.device}  VID:PID={vid}:{pid}  desc={p.description!r}  hwid={p.hwid!r}")
