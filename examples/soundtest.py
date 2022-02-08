import analyse
import numpy
import pyaudio
 
pyaud = pyaudio.PyAudio()
stream = pyaud.open(format = pyaudio.paInt16,channels = 1,rate=44100,input_device_index=2,input=True)
 
while True:
    raws=stream.read(1024, exception_on_overflow = False)
    samples= numpy.fromstring(raws, dtype=numpy.int16)
    loudness = analyse.loudness(samples)
    if loudness > -15:
        print("Really loud")
