import ctypes
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok

# Create a handle for the scope
chandle = ctypes.c_int16()

# Attempt to open the PicoScope
status_open = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, 1)

try:
    assert_pico_ok(status_open)
    print("PicoScope connected successfully.")
except:  # Handle specific power issues or errors
    if status_open == 286 or status_open == 282:  # Power source errors
        ps.ps5000aChangePowerSource(chandle, status_open)
        print("Power source updated. Please retry.")
    else:
        print(f"Failed to connect to PicoScope. Error code: {status_open}")

# Close the PicoScope if successfully opened
if chandle.value:
    status_close = ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status_close)
    print("PicoScope closed successfully.")
