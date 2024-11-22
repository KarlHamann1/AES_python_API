# Arduino Side-Channel Data Acquisition

This project demonstrates how to capture side-channel data, such as power traces or LED brightness changes, from an Arduino performing AES encryption. It uses a photodiode connected to a Picoscope for capturing the data and a Python-based interface for communication and control.

---

## **Features**
- Arduino-based AES encryption.
- Data acquisition using a Picoscope triggered by the Arduino's hardware pin.
- Python interface for controlling the Arduino and capturing data.
- Automatic saving of captured traces in `.npy` format for efficient storage.
- Metadata (e.g., timestamps, sampling rates) included for easier post-processing.
- Visualization of captured traces with Matplotlib.

### **Hardware**
- **Arduino Uno** with AES encryption firmware.
- **Photodiode** connected to a **transimpedance amplifier (TIA)**.
- **Picoscope** (e.g., Picoscope 5000 series).
- Computer with USB connections for Arduino and Picoscope.

### **Software**
- Python 3.8+ with the following libraries:
  - `numpy`
  - `matplotlib`
  - `picosdk`
- Arduino IDE for uploading the firmware.

---

## **Setup Instructions**

### **1. Arduino Setup**
1. Upload the `AES_Encipher.ino` firmware to the Arduino Uno.
2. Ensure the Arduino is connected to the computer via USB.
3. Configure a digital pin (e.g., A0) to output a trigger signal during encryption.

### **2. Picoscope Setup**
1. Connect the photodiode output to the Picoscope input (Channel A).
2. Connect the Arduino trigger pin to the Picoscope’s external trigger input (EXT).
3. Ensure the Picoscope is connected to the computer via USB.

### **3. Python Environment**
1. Install Python 3.8+.
2. Install the required libraries:
   ```bash
   pip install numpy matplotlib picosdk

### **4. Run the Project**
- python main.py

View and analyze the saved traces using the visualization script:

- python visualize.py data/<filename>.npy
