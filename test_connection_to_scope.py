import ctypes
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok

# Enumerate devices
count = ctypes.c_int16(0)
serials = ctypes.create_string_buffer(256)
serial_length = ctypes.c_int16(256)

status_enum = ps.ps5000aEnumerateUnits(
    ctypes.byref(count), 
    ctypes.cast(serials, ctypes.POINTER(ctypes.c_char)), 
    ctypes.byref(serial_length)
)

if status_enum == 0:  # PICO_OK
    if count.value == 0:
        print("No devices found.")
        exit()
    print(f"Devices found: {count.value}")
    print(f"Serial numbers: {serials.value.decode('utf-8')}")
else:
    print(f"Device enumeration failed with code: {status_enum}")
    exit()

# Create a handle for the device
chandle = ctypes.c_int16()
PS5000A_DR_12BIT = 1  # 12-bit resolution

# Attempt to open the device
status_openunit = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, PS5000A_DR_12BIT)

# Handle USB compatibility issue
if status_openunit == ps.PICO_USB3_0_DEVICE_NON_USB3_0_PORT:
    print("Device connected to a USB 2.0 port or running in USB 2.0 compatibility mode.")
    print("Attempting to configure power source...")
    status_changepower = ps.ps5000aChangePowerSource(chandle, ps.PICO_USB3_0_DEVICE_NON_USB3_0_PORT)
    assert_pico_ok(status_changepower)
    print("Power source configured successfully. Retrying to open the device...")
    # Retry opening the device
    status_openunit = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, PS5000A_DR_12BIT)

# Check if the device opened successfully
assert_pico_ok(status_openunit)
print("PicoScope connected successfully.")

# Close the PicoScope
status_close = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status_close)
print("PicoScope closed successfully.")
