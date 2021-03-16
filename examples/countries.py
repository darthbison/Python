from openpyxl import load_workbook
import os,sys
import re

wb = load_workbook(filename = 'Countries.xlsx')
sheet = wb.active

def extract_word(text):
    regex = r"(\w|\s)*"
    matches = re.finditer(regex, text, re.DOTALL)
    newstr = ''
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        newstr = newstr + match.group()
    return newstr


def gp(directory):
        fileName=directory + "countries.txt"
        cells = sheet['A2': 'F281']
        for a1 in cells:
          data = ""
          for x in range(6):
             value = ""
             if x == 3 :
               text = extract_word(str(a1[x].value))
               valuebytes = text.encode("utf-8")
               value = valuebytes.decode("utf-8")
             else :
               value = str(a1[x].value)
             data += value + "\t"
          data += "\n"
          with open(fileName, "a") as myfile:
             myfile.write(data)

if __name__ == "__main__":
    gp("")
