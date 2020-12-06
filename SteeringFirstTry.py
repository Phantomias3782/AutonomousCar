import RPi.GPIO as IO

IO.setwarnings(False)
IO.setmode(IO.BOARD)
IO.setup(35,IO.OUT)

s=IO.PWM(35,250)

full_left=41
straight=33
full_right=20


s.start(0)

s.ChangeDutyCycle(straight)
input("mitte")
s.ChangeDutyCycle(full_left)
input("links")
s.ChangeDutyCycle(full_right)
input("rechts")
s.ChangeDutyCycle(0)
s.stop()