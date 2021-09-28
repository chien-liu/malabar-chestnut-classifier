
const int in2Pin=7;
const int in1Pin=8;
const int enPin=9;
char state=0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(in2Pin,OUTPUT); //in2
  pinMode(in1Pin,OUTPUT); //in1
  pinMode(enPin,OUTPUT); //en
  state='R';
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0){
    state=Serial.read();
  }
  Serial.println(state);
  switch(state){
    case 'L':
      Serial.println("State L");
      digitalWrite(in1Pin,HIGH);
      digitalWrite(in2Pin,LOW);
      analogWrite(enPin, 200);
      delay(500);
      state='E';
      break;
    case 'R':
      digitalWrite(in1Pin,LOW);
      digitalWrite(in2Pin,HIGH);
      analogWrite(enPin, 200);
      delay(500);
      state='E';
      break;
    case 'E':
      analogWrite(enPin, 0);
      break;
  }
 //delay(500);
}
