import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc

# Enumerate devices
count = ctypes.c_int16(0)  # 16-bit signed integer to hold device count
serials = ctypes.create_string_buffer(256)  # Buffer for serial numbers
serial_length = ctypes.c_int16(256)  # Buffer length as 16-bit integer

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
status = {}
PS5000A_DR_12BIT = 1  # 12-bit resolution

# Open the PicoScope with resolution
status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, PS5000A_DR_12BIT)
assert_pico_ok(status["openunit"])

# Set up Channel A
channelA = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
chARange = ps.PS5000A_RANGE["PS5000A_20V"]
status["setChA"] = ps.ps5000aSetChannel(chandle, channelA, 1, coupling_type, chARange, 0)
assert_pico_ok(status["setChA"])

# Get the maximum ADC count
maxADC = ctypes.c_int16()
status["maxADC"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maxADC"])

# Set up a trigger on Channel A
threshold = int(mV2adc(500, chARange, maxADC))
status["trigger"] = ps.ps5000aSetSimpleTrigger(
    chandle, 1, channelA, threshold,
    ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"], 0, 1000
)
assert_pico_ok(status["trigger"])

# Set up pre and post trigger samples
preTriggerSamples = 2500
postTriggerSamples = 2500
maxSamples = preTriggerSamples + postTriggerSamples

# Get timebase information
timebase = 8
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int32()
status["getTimebase"] = ps.ps5000aGetTimebase2(
    chandle, timebase, maxSamples,
    ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), 0
)
assert_pico_ok(status["getTimebase"])

# Start block capture
status["runBlock"] = ps.ps5000aRunBlock(
    chandle, preTriggerSamples, postTriggerSamples,
    timebase, None, 0, None, None
)
assert_pico_ok(status["runBlock"])

# Wait for the capture to complete
ready = ctypes.c_int16(0)
while ready.value == 0:
    status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))

# Create a buffer for data
bufferA = (ctypes.c_int16 * maxSamples)()

# Set data buffer for Channel A
status["setDataBuffersA"] = ps.ps5000aSetDataBuffer(
    chandle, channelA, ctypes.byref(bufferA), maxSamples,
    0, ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]
)
assert_pico_ok(status["setDataBuffersA"])

# Retrieve data
overflow = ctypes.c_int16()
cmaxSamples = ctypes.c_int32(maxSamples)
status["getValues"] = ps.ps5000aGetValues(
    chandle, 0, ctypes.byref(cmaxSamples), 0,
    ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"], 0,
    ctypes.byref(overflow)
)
assert_pico_ok(status["getValues"])

# Convert ADC data to mV
adc2mVChA = adc2mV(bufferA, chARange, maxADC)

# Create time data
time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

# Plot data
plt.plot(time, adc2mVChA[:], label="Channel A")
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.title('PicoScope 5244D - Channel A')
plt.legend()
plt.show()

# Stop the PicoScope
status["stop"] = ps.ps5000aStop(chandle)
assert_pico_ok(status["stop"])

# Close the PicoScope
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])

# Print status
print(status)
