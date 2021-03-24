import sys
import urllib.request
import xml.etree.ElementTree as ET
import re

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from Hologram.HologramCloud import HologramCloud

if __name__ == "__main__":

    verseContent = ""
    with urllib.request.urlopen('https://www.biblegateway.com/usage/votd/rss/votd.rdf?31') as response:
        verseContent = response.read()


    root = ET.fromstring(verseContent)
    items = []


    for child in root:
        #print(child.tag)
        items = child[10]

    chapter = items[0].text
    chapterContent = items[3].text

    totalMessage = chapter + "\n\n" + chapterContent.strip()
    totalMessage =  re.sub('&[rl]dquo;', '"', totalMessage)
    totalMessage = re.sub('<br/><br/>', '\n', totalMessage)


    device_key = "" 

    credentials = {'devicekey': device_key}

    hologram = HologramCloud(credentials, network='cellular')

    result = hologram.network.connect()
    if result == False:
        print('Failed to connect to cell network')

    print('Cloud type: ' + str(hologram))

    print('Network type: ' + str(hologram.network_type))

    recv = hologram.sendMessage(totalMessage,
                                topics = ["TOPIC1","TOPIC2"],
                                timeout = 3)

    print('RESPONSE MESSAGE: ' + hologram.getResultString(recv))

    print('LOCAL IP ADDRESS: ' + str(hologram.network.localIPAddress))
    print('REMOTE IP ADDRESS: ' + str(hologram.network.remoteIPAddress))

    hologram.network.disconnect()

