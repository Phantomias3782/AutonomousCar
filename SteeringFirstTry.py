import RPi.GPIO as IO
import time
import os, sys
import picam

IO.setwarnings(False)
IO.setmode(IO.BOARD)
IO.setup(11,IO.OUT)
IO.setup(15,IO.OUT)

t=IO.PWM(11,100)
s=IO.PWM(15,100)


'''
throttle and steering voltages:

Full Fwd: .?
Idle: .?
Full Reverse: .?

Full Left: .?
Fwds: .?
Full Right: .?
'''
full_fwd=19
still_fwd=28
full_rwd=36

full_left=19
straight=28
full_right=33.5

t.start(still_fwd)
s.start(straight)
input("S: 0 G:0")

t.ChangeDutyCycle(still_fwd)
s.ChangeDutyCycle(straight)
input("S: 0 G:0")

t.ChangeDutyCycle(full_fwd)
s.ChangeDutyCycle(full_left)
input("S: L G:1")

t.ChangeDutyCycle(full_rwd/2)
s.ChangeDutyCycle(full_right/2)
input("S: R/2 G:-1/2")

t.ChangeDutyCycle(full_rwd)
s.ChangeDutyCycle(full_right)
input("S: R G:-1")