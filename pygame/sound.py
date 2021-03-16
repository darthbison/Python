import pygame, sys, time 
from pygame.locals import * 
 
pygame.init() 
#DISPLAYSURF = pygame.display.set_mode((400, 300)) 
#pygame.display.set_caption('Hello World!')
soundObj = pygame.mixer.Sound('beeps.wav') 
soundObj.play() 
time.sleep(1) # wait and let the sound play for 1 second 
soundObj.stop() 
