import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import time
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import paho.mqtt.client as mqtt


app = FastAPI()


model = tf.keras.models.load_model("glaucoma_model.h5")


TRIG = 23  
ECHO = 24  
RED_LED = 17  
GREEN_LED = 27  

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)

MQTT_BROKER = "mqtt.example.com"
MQTT_TOPIC = "glaucoma/detection"
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

def capture_image():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        cv2.imwrite("eye_image.jpg", frame)
    camera.release()
    return "eye_image.jpg"

def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    
    distance = (pulse_end - pulse_start) * 17150
    return distance

def predict_glaucoma(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  
    img = np.reshape(img, (1, 128, 128, 1))
    prediction = model.predict(img)
    return prediction[0][0] > 0.5  

@app.post("/detect")
async def detect_glaucoma(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    result = predict_glaucoma(file_path)
    
    if result:
        GPIO.output(RED_LED, True)
        GPIO.output(GREEN_LED, False)
        client.publish(MQTT_TOPIC, "Glaucoma Detected")
        return {"result": "Glaucoma Detected"}
    else:
        GPIO.output(RED_LED, False)
        GPIO.output(GREEN_LED, True)
        client.publish(MQTT_TOPIC, "No Glaucoma Detected")
        return {"result": "No Glaucoma Detected"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
