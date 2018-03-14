#include <XBee.h>
#include <Servo.h>
#include <SPI.h>
#include <Wire.h> 
#include "Utils.h"
#include "PX4Flow.h"
//Setting up IMU 
volatile int x = 0;
const int ESC_PIN = 9;
const int STEERING_SERVO_PIN = 5;
const int STEERING_ANGLE_STEP = 10;
const int THROTTLE_STEP = 1;
Servo esc;
Servo steeringServo;
XBee xbee = XBee();
bool status;
unsigned long start = millis();
float throttle;       // Throttle value between 0 and 180.
float steeringAngle;  // Steering angle between 0 and 180.
// global constants for 9 DoF fusion and AHRS (Attitude and Heading Reference System)
float steering;
XBeeAddress64 addr64 = XBeeAddress64(0x0013a200, 0x4108214A);
TxStatusResponse txStatus = TxStatusResponse();

uint8_t* data;
//float* convertedFloats;

int numFloats=8;
//float toSend[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

//uint8_t* payload;
float* floatValues;
float x_error, y_error, z_error;
void setup() {
  Wire.begin();
  Serial.begin(57600);
  xbee.setSerial(Serial);
  esc.attach(ESC_PIN);
  steeringServo.attach(STEERING_SERVO_PIN);
  Serial.flush();
  throttle = 0;
  steeringAngle = 90;
  issueCommands();
  averageAccelerationX();
  averageAccelerationY();
  //z_error = averageAccelerationZ();
  //calcNoise();
  //initPayload(numFloats * 4);
  
}

int option = 0;
int lengthOfRXPacket = 2;
XBeeResponse response = XBeeResponse();
// create reusable response objects for responses we expect to handle 
Rx16Response rx16 = Rx16Response();
Rx64Response rx64 = Rx64Response();
float yaw_1=0;
float gz_1=0;
float roll_1=0;
float pitch_1=0;
float xVelocityImu = 0;
float yVelocityImu = 0;
float zVelocityImu = 0;
float xPosition;
float yPosition;
float xVelocity;
float yVelocity;
long last_check = 0;
int px = 0;
int py = 0;
float focal_length_px = (3.6) / (4.0f * 6.0f) * 1000.0f;
PX4Flow sensor = PX4Flow(); 
int counter=0;
void loop() {

      if (xbee.readPacket(5000)) {
          // Got a rx packet.
          if (xbee.getResponse().getApiId() == RX_16_RESPONSE || xbee.getResponse().getApiId() == RX_64_RESPONSE) {
            
            if (xbee.getResponse().getApiId() == RX_16_RESPONSE) {
              xbee.getResponse().getRx16Response(rx16);
              option = rx16.getOption();
              data = rx16.getData();
            } else {
              xbee.getResponse().getRx64Response(rx64);
              option = rx64.getOption();
              data = rx64.getData();
            }
            // Received RX packet.
            float* convertedFloats = Utils::convert_to_floats(data, lengthOfRXPacket);
            // Extract values 
            throttle = convertedFloats[0];
            steering = convertedFloats[1];
            counter++;
            Serial.print("counter:");
            Serial.println(counter);
            Serial.print("steering:");
            Serial.println(steering);
            throttle = constrain(throttle, 0, 120);
            steeringAngle = (steering*180/3.14)+90;
            steeringAngle = constrain(steeringAngle, 65,115);
            issueCommands();

            free(convertedFloats);
          }
      }
      //Serial Communication for IMU
      Wire.beginTransmission(104);
      imu();
      yaw_1 = getYaw();
      gz_1 = getGz();
      roll_1 = getRoll();
      pitch_1 = getPitch();
      xVelocityImu = getVx();
      yVelocityImu = getVy();
      zVelocityImu = getVz();
      Wire.endTransmission();
      //change toSend to send the data you want to send. The IMU velocities are named xVelocityImu, yVelocityImu, zVelocityImu.
      
      float toSend[] = {getAx(), getAx_e(), getAy(), getAy_e(), xVelocityImu, yVelocityImu, steering,steeringAngle };
      uint8_t* payload = Utils::convert_to_bytes(toSend, 8);
      // Create TX packet.
      // Send TX Packet.
      Tx64Request tx = Tx64Request(addr64, payload, numFloats * 4);
      //xbee.send(tx);
      delay(100);
      free(payload);
}

// union to convery float to byte string
union floatContainer {
    uint8_t b[4];
    float fval;
} floatConverter;

// count = number of floats represented by
// data array.
float* readFloat(uint8_t data[], int count){
  float values [count];
  for(int i = 0; i < count; i++){
    for (int j = 0; j < 4; j++) {
      floatConverter.b[j] = data[i*4 + j];
    }
    values[i] = floatConverter.fval;
  }
  return values;
} 
void issueCommands() {
  esc.write(throttle);
  steeringServo.write(steering);
}


