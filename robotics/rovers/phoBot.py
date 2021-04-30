#!/usr/bin/env python

import threading
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.template
import time
import requests


#Initialize TOrnado to use 'GET' and load index.html
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        loader = tornado.template.Loader(".")
        self.write(loader.load("/opt/apps/BrowserBot/rover_client.html").generate())

#Code for handling the data sent from the webpage
class WSHandler(tornado.websocket.WebSocketHandler):
    
    def open(self):
        print ("connection opened...")
        
    def check_origin(self,origin):
        return True
    def on_message(self, message):      # receives the data from the webpage and is stored in the variable message
        
        accessToken = ""
        deviceID = ""
        controlURL = "https://api.spark.io/v1/devices/" + deviceID + "/control"
        
        # prints the revived from the webpage 
        if message == "u":                # checks for the received data and assigns different values to c which controls the movement of robot.
            c = "8"
        if message == "d":
            c = "2"
        if message == "l":
            c = "6"
        if message == "r":
            c = "4"
        if message == "s":
            c = "5"
        
        if c == '8' :
            requests.post(controlURL, data = {'params':'F-100', 'access_token' : accessToken})
            
        elif c == '2' :
            requests.post(controlURL, data = {'params':'B-75', 'access_token' : accessToken})
            
        elif c == '6' :
            requests.post(controlURL, data = {'params':'L-50', 'access_token' : accessToken})
            
        elif c == '4' :
            requests.post(controlURL, data = {'params':'R-50', 'access_token' : accessToken})
        
        elif c == '5' :
            requests.post(controlURL, data = {'params':'S', 'access_token' : accessToken})
     
    def on_close(self):
        print ("connection closed...")

application = tornado.web.Application([
    (r'/ws', WSHandler),
    (r'/', MainHandler),
    (r"/(.*)", tornado.web.StaticFileHandler, {"path": "./resources"}),
])

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print ("Ready")
        while running:
            time.sleep(.2)              # sleep for 200 ms
        

if __name__ == "__main__":
    running = True
    thread1 = myThread(1, "Thread-1", 1)
    thread1.setDaemon(True)
    thread1.start()  
    application.listen(9093)          	#starts the websockets connection
    
    
    tornado.ioloop.IOLoop.instance().start()
  

