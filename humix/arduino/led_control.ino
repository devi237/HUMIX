// HUMIX LED Alert System
// Green  → Pin 8  (Normal)
// Yellow → Pin 9  (Warning: 75% threshold)
// Red    → Pin 10 (Alert: threshold exceeded)

const int GREEN  = 8;
const int YELLOW = 9;
const int RED    = 10;

void setup() {
  Serial.begin(9600);
  pinMode(GREEN,  OUTPUT);
  pinMode(YELLOW, OUTPUT);
  pinMode(RED,    OUTPUT);
  setLED('G'); // Start with green
}

void setLED(char cmd) {
  if (cmd == 'G') {
    digitalWrite(GREEN,  HIGH);
    digitalWrite(YELLOW, LOW);
    digitalWrite(RED,    LOW);
  } else if (cmd == 'Y') {
    digitalWrite(GREEN,  LOW);
    digitalWrite(YELLOW, HIGH);
    digitalWrite(RED,    LOW);
  } else if (cmd == 'R') {
    digitalWrite(GREEN,  LOW);
    digitalWrite(YELLOW, LOW);
    digitalWrite(RED,    HIGH);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'G' || cmd == 'Y' || cmd == 'R') {
      setLED(cmd);
    }
  }
}
