/*
 * ==============================================================================
 * INDUSTRIAL PPE SAFETY PLATFORM - ESP32 HARDWARE ACTUATOR FIRMWARE (v2.1)
 * Target Board: ESP32 DevKit (30-pin or 38-pin)
 *
 * Hardware Wiring:
 *   - GPIO 25 (Pin D25) --> Red LED Anode (+)
 *   - GPIO 26 (Pin D26) --> Yellow LED Anode (+)
 *   - GPIO 27 (Pin D27) --> Green LED Anode (+)
 *   - GPIO 14 (Pin D14) --> Buzzer Signal / Positive (+)
 *   - GND               --> Common Ground for All Cathodes / Negatives (-)
 *
 * Safety Logic & Rules:
 *   - Boot / Idle: All LEDs and Buzzer are completely OFF.
 *   - STATE:COMPLIANT --> ONLY Green LED ON. Buzzer is SILENT.
 *                         (Triggered ONLY when ALL 4 PPE items are worn!)
 *   - STATE:PENDING   --> ONLY Yellow LED ON. Buzzer is SILENT.
 *   - STATE:VIOLATION --> ONLY Red LED ON + Buzzer sounds!
 *                         (Triggered if any required PPE is missing, e.g. helmet-only)
 *   - STATE:IDLE      --> All LEDs and Buzzer completely OFF.
 *   - Polarity Support: Supports both Active-HIGH and Active-LOW buzzer modules.
 *                       Can be toggled in code or via serial command "POLARITY:LOW".
 * ==============================================================================
 */

// Pin Definitions
const int PIN_RED    = 25;
const int PIN_YELLOW = 26;
const int PIN_GREEN  = 27;
const int PIN_BUZZER = 14;

// Timing Configuration
const unsigned long BUZZER_DURATION_MS = 2500; // Buzzer sounds for 2.5s max per violation to avoid ear fatigue
unsigned long violationStartTime       = 0;
bool isViolationActive                 = false;

// -----------------------------------------------------------------------------
// BUZZER POLARITY CONFIGURATION
// -----------------------------------------------------------------------------
// Standard 3-pin buzzer breakout modules (VCC, GND, I/O) are ACTIVE-LOW (sound on LOW).
// Direct 2-pin passive/active buzzers are ACTIVE-HIGH (sound on HIGH).
// Default is set to true (ACTIVE-LOW) so the buzzer is SILENT during Green & Yellow,
// and ONLY sounds when a RED VIOLATION occurs!
// Can be toggled anytime from the dashboard UI button or serial command "POLARITY:HIGH".
bool buzzerActiveLow = true;

void buzzerOn() {
  if (buzzerActiveLow) {
    digitalWrite(PIN_BUZZER, LOW);
  } else {
    digitalWrite(PIN_BUZZER, HIGH);
  }
}

void buzzerOff() {
  if (buzzerActiveLow) {
    digitalWrite(PIN_BUZZER, HIGH);
  } else {
    digitalWrite(PIN_BUZZER, LOW);
  }
}

// -----------------------------------------------------------------------------
// Turn All Outputs Completely OFF
// -----------------------------------------------------------------------------
void allOff() {
  digitalWrite(PIN_RED, LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_GREEN, LOW);
  buzzerOff();
}

// -----------------------------------------------------------------------------
// Hardware Setup
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(30);

  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_YELLOW, OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // Initial State: All indicators completely OFF on power-up
  allOff();

  Serial.println("\n[BOOT READY] ESP32 PPE Safety Actuator Online (v2.1).");
  Serial.println("[PINS] RED=GPIO25 | YELLOW=GPIO26 | GREEN=GPIO27 | BUZZER=GPIO14");
  Serial.print("[BUZZER MODE] ");
  Serial.println(buzzerActiveLow ? "ACTIVE-LOW (LOW=Sound, HIGH=Mute)" : "ACTIVE-HIGH (HIGH=Sound, LOW=Mute)");
}

// -----------------------------------------------------------------------------
// Main Loop
// -----------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  // Auto-silence buzzer after BUZZER_DURATION_MS to prevent continuous ear-splitting noise
  // even while the Red LED remains ON to mark the active violation.
  if (isViolationActive && (now - violationStartTime > BUZZER_DURATION_MS)) {
    buzzerOff();
  }

  // Read Commands from Python over USB Serial
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.length() == 0) return;

    // 🔴 1. Confirmed Violation (ONLY Red ON + Buzzer sounds)
    if (command.startsWith("STATE:VIOLATION")) {
      isViolationActive  = true;
      violationStartTime = millis();

      digitalWrite(PIN_GREEN, LOW);
      digitalWrite(PIN_YELLOW, LOW);
      digitalWrite(PIN_RED, HIGH);
      buzzerOn();

      Serial.println("[ACK] STATE:VIOLATION -> RED ON, BUZZER SOUNDING");
    }
    // 🟡 2. Warning / Verifying (ONLY Yellow ON, Buzzer is SILENT)
    else if (command == "STATE:PENDING") {
      isViolationActive = false;

      digitalWrite(PIN_RED, LOW);
      digitalWrite(PIN_GREEN, LOW);
      digitalWrite(PIN_YELLOW, HIGH);
      buzzerOff();

      Serial.println("[ACK] STATE:PENDING -> YELLOW ON, BUZZER SILENT");
    }
    // 🟢 3. Fully Compliant Worker (ONLY Green ON, Buzzer is SILENT)
    // ONLY triggered when ALL 4 PPE items (Helmet, Vest, Boots, Gloves) are present!
    else if (command == "STATE:COMPLIANT") {
      isViolationActive = false;

      digitalWrite(PIN_RED, LOW);
      digitalWrite(PIN_YELLOW, LOW);
      digitalWrite(PIN_GREEN, HIGH);
      buzzerOff();

      Serial.println("[ACK] STATE:COMPLIANT -> GREEN ON (ALL 4 PPE VERIFIED), BUZZER SILENT");
    }
    // ⚪ 4. Idle / No One in View (ALL LEDs OFF, Buzzer is SILENT)
    else if (command == "STATE:IDLE") {
      isViolationActive = false;
      allOff();

      Serial.println("[ACK] STATE:IDLE -> ALL OFF");
    }
    // ⚙️ 5. Runtime Polarity Commands
    else if (command == "POLARITY:LOW" || command == "BUZZER:ACTIVE_LOW") {
      buzzerActiveLow = true;
      buzzerOff();
      Serial.println("[CONFIG] Buzzer Polarity set to ACTIVE-LOW");
    }
    else if (command == "POLARITY:HIGH" || command == "BUZZER:ACTIVE_HIGH") {
      buzzerActiveLow = false;
      buzzerOff();
      Serial.println("[CONFIG] Buzzer Polarity set to ACTIVE-HIGH");
    }
    // 🛠️ 6. Hardware Diagnostics
    else if (command == "TEST:BUZZER_ON") {
      buzzerOn();
      Serial.println("[TEST] Buzzer forced ON");
    }
    else if (command == "TEST:BUZZER_OFF") {
      buzzerOff();
      Serial.println("[TEST] Buzzer forced OFF");
    }
  }
}
