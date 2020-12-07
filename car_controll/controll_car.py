import RPi.GPIO as IO
import math

class Car():
    def __init__(self):
        #setup the raspberry for PWM on Pin 35
        IO.setwarnings(False)
        IO.setmode(IO.BOARD)
        IO.setup(35,IO.OUT)

        #assing pin 35 to steering
        self.steering =IO.PWM(35,250)

        #define max values
        self.FULL_LEFT=41
        self.STRAIGHT=33
        self.FULL_RIGHT=20
        
        #set steering dutycycle to 0
        self.steering.start(0)

    def __del__(self):
        self.steering.stop()

    def steer(self,steeringrate):
        error=False
        if steeringrate == 0:
            self.steerig.ChangeDutyCycle(self.STRAIGHT)
        elif steeringrate > 0 and steeringrate <= 1:
            self.steerig.ChangeDutyCycle(self.FULL_LEFT * steeringrate)
        elif steeringrate < 0 and steeringrate >= -1:
            steeringrate = abs(steeringrate)
            self.steerig.ChangeDutyCycle(self.FULL_RIGHT * steeringrate)
        time.sleep(1)
        #fix twitching of servo with resetting the dc to 0
        self.steerig.ChangeDutyCycle(0)
        if error:
            return False
        return True