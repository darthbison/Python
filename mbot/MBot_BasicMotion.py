# generated by mBlock5 for CyberPi
# codes make you happy

import event, time, cyberpi, mbot2
import time

@event.start
def on_start():
    # Moving function display
    #
    # Upload mode and Live mode are supported.
    cyberpi.console.clear()
    cyberpi.console.set_font(12)
    cyberpi.console.println("Hello Silicon 100 Mentees!!")
    cyberpi.audio.play('hi')

@event.is_press('up')
def is_joy_press():
    time.sleep(1)
    cyberpi.audio.play('beeps')
    mbot2.straight(-10)

@event.is_press('down')
def is_joy_press1():
    time.sleep(1)
    cyberpi.audio.play('beeps')
    mbot2.straight(10)

@event.is_press('right')
def is_joy_press2():
    time.sleep(1)
    cyberpi.audio.play('beeps')
    mbot2.turn(90)

@event.is_press('left')
def is_joy_press3():
    time.sleep(1)
    cyberpi.audio.play('beeps')
    mbot2.turn(-90)

