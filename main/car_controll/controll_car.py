import RPi.GPIO as IO
import time

class Car():
    def __init__(self):
        #setup the raspberry for PWM on Pin 33+35
        IO.setwarnings(False)
        IO.setmode(IO.BOARD)
        IO.setup(33,IO.OUT)
        IO.setup(35,IO.OUT)
        #assing pin 33 to throttle and 35 to steering
        self.throttle =IO.PWM(33,50)
        self.throttle.start(0)

        self.steering =IO.PWM(35,50)
        self.steering.start(0)
        
        #steering config
        self.FULL_LEFT=10#44
        self.STRAIGHT=8
        self.FULL_RIGHT=6#19
        self.LEFT=self.FULL_LEFT-self.STRAIGHT
        self.RIGHT=self.STRAIGHT-self.FULL_RIGHT

    def __del__(self):
        self.throttle.ChangeDutyCycle(0.1)
        self.steering.stop()
        self.throttle.stop()

    def steer(self,steeringrate):
        if steeringrate == None:
            pass
        else:
            if abs(steeringrate) >2:
                steeringrate=0
            elif steeringrate < -1:
                 steeringrate =-1
            elif steeringrate > 1:
                steeringrate=1 
            else:
                 steeringrate=steeringrate*1.99999999953251
            if steeringrate == 0:
                self.steering.ChangeDutyCycle(self.STRAIGHT)    
            elif steeringrate > 0:
                steeringrate=self.STRAIGHT + self.LEFT * steeringrate            
                if steeringrate > self.FULL_LEFT:
                    steeringrate=self.FULL_LEFT
                self.steering.ChangeDutyCycle(steeringrate)
            elif steeringrate < 0:
                steeringrate=self.STRAIGHT - self.RIGHT * abs(steeringrate)
                if steeringrate < self.FULL_RIGHT:
                    steeringrate=self.FULL_RIGHT
                self.steering.ChangeDutyCycle(steeringrate)
            else:
                print("Steeringinput incorrect")

    def run(self):
        self.throttle.ChangeDutyCycle(6.7)

    def stop(self):
        self.throttle.ChangeDutyCycle(0.1)
