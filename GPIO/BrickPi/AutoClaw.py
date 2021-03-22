from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import time     # import the time library for the sleep function
import brickpi3 # import the BrickPi3 drivers
import sys      # import sys for sys.exit()
import numpy as np

class AutoClaw:
    def __init__(self):
        self.BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.
        self.PORT_MOTOR_CLAW = self.BP.PORT_A
        self.threshold = 0
        self.hasItem = False
        #self.runClaw()
    
    def runClaw(self):
        self.BP.offset_motor_encoder(PORT_MOTOR_CLAW, BP.get_motor_encoder(PORT_MOTOR_CLAW))
        self.BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.NXT_LIGHT_ON)
        while True:
            try:
                    lightvalue = self.BP.get_sensor(BP.PORT_1)

                    if self.threshold == 0:
                        self.threshold = getLightAverage(lightvalue)

                    # print("Threshold: " + str(threshold))
                    # print("LV: " + str(lightvalue))

                    
                    if lightvalue >= self.threshold:
                        self.hasItem = False

                    if self.hasItem == False:
                        if lightvalue < self.threshold:
                            self.BP.set_motor_power(self.PORT_MOTOR_CLAW , 25)
                            time.sleep(2)
                            self.hasItem = True
                        elif lightvalue > self.threshold:
                            self.BP.set_motor_power(self.PORT_MOTOR_CLAW , -25)
                        
                    if self.hasItem & lightvalue < self.threshold:
                        self.BP.set_motor_power(self.PORT_MOTOR_CLAW , -25)

            except brickpi3.SensorError as error:
                    print(error)

    def getLightAverage(lightvalue):
        count = 0
        lightValues = []
        while count <= 10:
            if count == 10:
                return np.average(lightValues) - 200  
            else:  
                lightValues.append(lightvalue)
                count = count + 1


# try:
#         while True:
#             try:
                
#                 lightvalue = BP.get_sensor(BP.PORT_1)

#                 if threshold == 0:
#                     threshold = getLightAverage(lightvalue)

#                 # print("Threshold: " + str(threshold))
#                 # print("LV: " + str(lightvalue))

                
#                 if lightvalue >= threshold:
#                     hasItem = False

#                 if hasItem == False:
#                     if lightvalue < threshold:
#                         BP.set_motor_power(PORT_MOTOR_CLAW , 25)
#                         time.sleep(2)
#                         hasItem = True
#                     elif lightvalue > threshold:
#                         BP.set_motor_power(PORT_MOTOR_CLAW , -25)
                    
#                 if hasItem & lightvalue < threshold:
#                     BP.set_motor_power(PORT_MOTOR_CLAW , -25)

#             except brickpi3.SensorError as error:
#                 print(error)
# except KeyboardInterrupt: # except the program gets interrupted by Ctrl+C on the keyboard.
#         BP.reset_all()

    
