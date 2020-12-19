import RPi.GPIO as IO
import time

class Car():
    def __init__(self):
        #setup the raspberry for PWM on Pin 35
        IO.setwarnings(False)
        IO.setmode(IO.BOARD)
        IO.setup(35,IO.OUT)

        #assing pin 35 to steering


        self.steering =IO.PWM(35,50)
        self.steering.start(0)
        '''
        #define max values 250hz
        self.FULL_LEFT=39#44
        self.STRAIGHT=32.5
        self.FULL_RIGHT=25#19
        '''
        #200HZ
        self.FULL_LEFT=10#44
        self.STRAIGHT=8
        self.FULL_RIGHT=6#19
        self.LEFT=self.FULL_LEFT-self.STRAIGHT
        self.RIGHT=self.STRAIGHT-self.FULL_RIGHT
        #set steering dutycycle to 0
        

    def __del__(self):
        self.steering.stop()

    def steer(self,steeringrate):
        if steeringrate == None:
            pass
        #print("Steering None")    
        else:
            if abs(steeringrate) >2:
                steeringrate=0
            elif steeringrate < -1:
                 steeringrate =-1
            elif steeringrate > 1:
                steeringrate=1 
            else:
                 steeringrate=steeringrate*1.3
            if steeringrate == 0:
    #            print("input 0")
                self.steering.ChangeDutyCycle(self.STRAIGHT)    
            elif steeringrate > 0:
                #print("input left")
    #            print(self.STRAIGHT + self.LEFT * steeringrate)
                steeringrate=self.STRAIGHT + self.LEFT * steeringrate            
                if steeringrate > self.FULL_LEFT:
                    steeringrate=self.FULL_LEFT
                self.steering.ChangeDutyCycle(steeringrate)
            elif steeringrate < 0:
                #print("input right")
                steeringrate=self.STRAIGHT - self.RIGHT * abs(steeringrate)
                if steeringrate < self.FULL_RIGHT:
                    steeringrate=self.FULL_RIGHT
                self.steering.ChangeDutyCycle(steeringrate)
            else:
                print("Steeringinput ERROR!!!")
        #self.steering.stop()
#        time.sleep(0.5)
 #       self.steering.ChangeDutyCycle(0)

car = Car()
car.steer(0)
time.sleep(1)


'''
car.steer(1)
time.sleep(10)
car.steer(-1)
time.sleep(1)
car.steer(0)
time.sleep(1)
'''



car.steering.stop()
