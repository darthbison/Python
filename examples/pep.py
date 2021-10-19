from openpyxl import load_workbook
import os,sys
import random

wb = load_workbook(filename = 'pep.xlsx')
sheet = wb.active

def gp(directory):
        
        firstPhrase = []
        secondPhrase = []
        thirdPhrase = []
        fourthPhrase = []

        fullPhrase = ""
        
        cells = sheet['A1': 'D18']
        for cell in cells:
          
          for x in range(4):
             
             if x == 0 :
               firstPhrase.append(str(cell[x].value))
             elif x == 1:
               secondPhrase.append(str(cell[x].value))
             elif x == 2:
               thirdPhrase.append(str(cell[x].value))
             elif x == 3:
               fourthPhrase.append(str(cell[x].value))
             
        fullPhrase += random.choice(firstPhrase) + " " + random.choice(secondPhrase) + " " + random.choice(thirdPhrase) + " " + random.choice(fourthPhrase)
        print(fullPhrase)
          



if __name__ == "__main__":
    gp("")
