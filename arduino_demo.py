'''Demo with arduino and meshanism.
Device: Computer
    Dependency:
        1. numpy, opencv-python, pyserial, tensorflow-gpu(recommended) or tensorflow
        2. './net/vgg19.py

Connecting Device: 
    1. Arduino Uno
       Code of Arduino Uno is in './../arduino/motor_control/motor_control.ino'.
    2. Webcam

Features:
    1. Get an image from webcam.
    2. Analyse the image with VGG19 Net, check the location of 種子芽點.
    3. Send different instructions to Arduino Uno based on the result of analysis.
    4. Wait until Arduino Uno returns a messenge. Then start the next loop from step 1.

Communications Protocol with Arduino Uno:
    Serial
    (1) [Send instruction to Arduino Uno]
        Each instruction is consisted of 3 characters, e.g. "sre", "sle", "see".
        First character is set to be "s", which represents <START> of an instruction.
        Last (third) character is set to be "e", which represents <END> of an instruction.
        There is three choices of the second character. Either 'r', 'l', or 'e'.
            'r' means to move the motor clockwise.
            'l' means to move the motor counter-clockwise.
            'e' means not to move the motor, which we don't really use it.
    (2) [Messenge returned from Arduino Uno]
        Return "Done" (4 characters, without "\n")

'''

import cv2
from net.vgg19 import VGG19
import serial

CAM_PORT = 1
SERIAL_PORT = '/dev/ttyACM0'

def check_img(cap, vgg19):

    # flush cam buffer (0.2 sec)
    for _ in range(5):
        cap.read()

    _, img = cap.read()
    label = vgg19.predict(img) # label = 'u', 'd', 'm' # 0.8s

    # Add highlight on webcam image.
    # Different colors corresponds to different labels.
    shape = (img.shape[1], img.shape[0])
    if label == 'd':
        cv2.rectangle(img,(0,0),shape,(0,255,0),20)
    elif label == 'u':
        cv2.rectangle(img,(0,0),shape,(0,0,255),20)
    elif label == 'm':
        cv2.rectangle(img,(0,0),shape,(0,100,100),20)

    # Display the resulting frame
    cv2.imshow('frame',img)
    cv2.waitKey(1)
    return label

def move_motor(ser, label):
    if label == 'd':
        messenge = b'see'
    else:
        messenge = b'sle'
    ser.write(messenge)     # write a string
    while True:
        line = ser.readline()
        if line:
            break
    print(line)

if __name__ == '__main__':
    # Initialize the serial.
    ser = serial.Serial(SERIAL_PORT, 115200, timeout= 0.5)  # open serial port
    print(ser.name)         # check which port was really used

    # Initialize the webcam.
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # turn the autofocus off
    cap.set(cv2.CAP_PROP_FOCUS, 20)         # set focus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # set image width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set image height

    # Initialize VGG19 Neural Network .
    vgg19 = VGG19(vgg19_npy_path='./weights/fine_tune_weight.npy')

    while True:
        label = check_img(cap, vgg19)
        move_motor(ser, label)
