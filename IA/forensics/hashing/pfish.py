#!/usr/bin/python

import logging
import time
import sys
import _pfish


if __name__ == '__main__':
  PFISH_VERSION = '1.0'

  #Turn on logging
  logging.basicConfig(filename='pFishLog.log',level=logging.DEBUG,format='%(asctime)s %(message)s')

  #Process command line args
  _pfish.ParseCommandLine()

  #Record start time
  startTime = time.time()

  logging.info('Welcome to p-fish version ' + PFISH_VERSION + ' ... New Scan Started')
  _pfish.DisplayMessage('Welcome to p-fish ... version ' + PFISH_VERSION)

  logging.info('System: ' + sys.platform)
  logging.info('Version: ' + sys.version)

  #Traverse the file system directories and hash the file
  filesProcessed = _pfish.WalkPath()
  #Record the end time and calculate duration

  endTime = time.time()
  duration = endTime - startTime
  logging.info('Files processed: ' + str(filesProcessed) )
  logging.info('Elapsed time: ' + str(duration) + ' seconds')
  logging.info('Program Terminated Normally')
  _pfish.DisplayMessage("Program End")

  
