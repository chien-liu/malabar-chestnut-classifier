/**
 * Device: Arduino Uno
 * Connecting Device:
 *                   1. Computer (via USB)
 *                   2. 馬達驅動晶片
 *                          Pin7 is connected to in2
 *                          Pin8 is connected to in1
 *                          Pin9 is connected to en
 *                          
 * Features:
 *     (1) Recieve instructions from computer with serial communication.
 *     (2) Then move the motor corresponding to the instruction.
 *     (3) Return a messenge to the computer after the motor is moved.
 * 
 * Communications Protocol with Computer:
 *     Serial
 *     (1) [Recieve instructions from computer]
 *         Each instruction from computer is consisted of 3 characters, e.g. "sre", "sle", "see".
 *         First character is set to be "s", which represents <START> of an instruction.
 *         Last (third) character is set to be "e", which represents <END> of an instruction.
 *         There is three choices of the second character. Either 'r', 'l', or 'e'.
 *             'r' means to move the motor clockwise.
 *             'l' means to move the motor counter-clockwise.
 *             'e' means not to move the motor, which we don't really use it.
 *     (2) [Return a messenge to the computer]
 *         Return "Done" (4 characters, without "\n")
 *         
**/

const int in2Pin=7;
const int in1Pin=8;
const int enPin=9;
char state=0;

// Modify baud rate
#define baud_rate 115200

// Variables for decoding incoming serial data
boolean get_data;

void reset_data() {
  get_data = false;
  state = 0;
}

int decode() {
  int incomingBytes[3] = {0, 0, 0};
  incomingBytes[0] = Serial.read();

  // Check if the first char is 's' <START>
  if (incomingBytes[0] == 115) { // 's' == 115
    // Read data
    for (int i = 1; i < 3; i++) {
      incomingBytes[i] = Serial.read();
    }
    // Check if the last char is 'e' <END>
    if (incomingBytes[2] == 101) { // 'e' == 101
      get_data = true; // Activate the get_data flag
      // Check the command is either rotate right, rotate left, or stop
      switch (incomingBytes[1]) {
        case 114: // 'r' == 114
          state='r';
          break;

        case 108: // 'l' == 108
          state='l';
          break;

        case 101: // 'e' == 101
          state='e';
          break;

        default:
          // Invalid instruction
          get_data = false; // Deactivate the get_data flag
          return 0;
      }
    }
    else {
      // The last charcter is not 'e';
      return 0;
    }
  }
  else {
    // The first character is not 's'
    return 0;
  }
  return 1;
} // decode()

void action() {
  switch(state){
    case 'l':
      digitalWrite(in1Pin,HIGH);
      digitalWrite(in2Pin,LOW);
      analogWrite(enPin, 200);
      delay(500);
      // Stop motor
      digitalWrite(in1Pin, LOW);
      digitalWrite(in2Pin, LOW);
      analogWrite(enPin, 0);
      break;
    case 'r':
      digitalWrite(in1Pin,LOW);
      digitalWrite(in2Pin,HIGH);
      analogWrite(enPin, 200);
      delay(500);
      // Stop motor
      digitalWrite(in1Pin, LOW);
      digitalWrite(in2Pin, LOW);
      analogWrite(enPin, 0);
      break;
    case 'e':
    // Stop motor
      digitalWrite(in1Pin, LOW);
      digitalWrite(in2Pin, LOW);
      analogWrite(enPin, 0);
      delay(500);
      break;
  }
} // action()

void setup() {
  Serial.begin(baud_rate); // opens serial port, sets data rate to 9600 bps
  pinMode(in2Pin,OUTPUT);  //in2
  pinMode(in1Pin,OUTPUT);  //in1
  pinMode(enPin,OUTPUT);   //en
}
void loop() {
  reset_data();
  if (Serial.available() >= 3) {
    decode();
  }
  if (get_data) {
    action();
    Serial.print("Done");
  }
}
