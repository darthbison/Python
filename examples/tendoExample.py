#!/bin/python

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tendo.lib 


me = SingleInstance() # will sys.exit(-1) if other instance is running

while true:
  print("Hello")
