from ctypes import c_int16, byref
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok
from picosdk.errors import PicoSDKCtypesError

PICO_OK = 0
PICO_POWER_SUPPLY_NOT_CONNECTED = 286

PS5000A_DR_12BIT = 2  # 12-bit resolution

def initialize_picoscope():
    handle = c_int16()
    resolution = PS5000A_DR_12BIT  # Use raw integer for resolution

    # Attempt to open the device
    status = ps.ps5000aOpenUnit(byref(handle), None, resolution)
    if status == PICO_POWER_SUPPLY_NOT_CONNECTED:
        print("Power supply not connected. Attempting to configure power...")
        status = ps.ps5000aChangePowerSource(handle, PICO_POWER_SUPPLY_NOT_CONNECTED)
        if status != PICO_OK:
            raise PicoSDKCtypesError(f"Failed to change power source, status: {status}")
        
        # Retry opening the device
        status = ps.ps5000aOpenUnit(byref(handle), None, resolution)

    if status != PICO_OK:
        raise PicoSDKCtypesError(f"PicoSDK returned {status} during ps5000aOpenUnit")

    print(f"Device opened successfully, handle: {handle.value}")
    return handle

def close_picoscope(handle):
    status = ps.ps5000aCloseUnit(handle)
    assert_pico_ok(status)
    print("Device closed successfully")

try:
    scope_handle = initialize_picoscope()
    # Further setup and data acquisition
finally:
    if 'scope_handle' in locals():
        close_picoscope(scope_handle)
