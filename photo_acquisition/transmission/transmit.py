import os
from pathlib import Path


import serial


PORT = "COM3"  # Change this to your actual COM port (check in Device Manager)
BAUD_RATE = 115200 # synchronize with the MCU
OUTPUT_DIR = Path("captured")
NUM_IMS = 10000


def listen():

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    ser = serial.Serial(PORT, BAUD_RATE, timeout=10)

    print("Listening for image data...")
    data = bytearray()
    image_coming = False

    for im_num in range(NUM_IMS):

        while True:

            if not image_coming:
                line = ser.readline().strip()

                if line == b'PHOTO_START':
                    image_coming = True
                    data.clear()
                    print("Receiving image...")

            else:
                chunk = ser.read(256)
                if not chunk:
                    break

                if b"PHOTO_END" in chunk:
                    chunk = chunk[: chunk.index(b"PHOTO_END")]
                    data += chunk
                    break
                else:
                    data += chunk

        print("Size", len(data))

        target_path = OUTPUT_DIR / Path( f"{im_num}.jpg" )
        if data:
            with open(str(target_path), "wb") as file:
                file.write(data)
            print(f"Image saved as {target_path}")
        else:
            print("No image received.")

        image_coming = False

    ser.close()


if __name__ == "__main__":
    listen()
