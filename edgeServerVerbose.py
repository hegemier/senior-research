from flask import Flask, request
import redis
app = Flask(__name__)
import json
import time
import random
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io

'''
Edge Server: Verbose Version
This server will print out all of the steps for classification, sentiment analysis, etc.
'''

def setContact(robotId, contact): #set robot contact state to current time
    key = str(robotId) + "contact"
    r.mset({key: str(contact)})

def getContact(robotId): #get robot latest contact state
    key = str(robotId) + "contact"
    print("fetching id " + str(robotId))
    print("which is " + str(r.get(key)))
    if r.get(key) == None:
        return 0.0
    return float(r.get(key))

def evaluateEvent(event = "NO_EVENT"):
    '''
    This function evaluates data that is returned with the environment image to reason about any potential
    issues or updates to the robot environment. New information on environment hazards is then returned to the
    sentiment function.

    The sentiment function also works with a JSON-based ruleset that can be modified and distributed to any edge server.
    This ruleset provides a knowledge base for what actions the edge or robot should take given the reported event.
    In this example, we have a stored action for if a robot loses contact (returned here as LOST_CONTACT).

    More on this can be seen in the processImageSentiment function. 
    '''
    infoList = []

    if event == "DAMAGE_DETECTED": #robot self-reports damage, no analysis is required.
        infoList.append(event)
        return infoList
    # no event, so check if any robot lost contact. Each time some robot checks in, we do this for all robots. 
    lost = False
    print("Scanning events")
    for id in robotIds:
        print("robot ids are: " + str(robotIds))
        key = str(id) + "contact"
        lastTimeSeen = getContact(id)
        if lastTimeSeen != 0.0: #robot may have not checked in yet
            currentSeen = float(time.time())
            print(str(id) + ": no contact for " + str(currentSeen-lastTimeSeen))
            if currentSeen - lastTimeSeen > 20: #robot has been disconnected for too long
                lost = True
                infoList.append("LOST_CONTACT")
                infoList.append(id)
    if len(infoList) == 0:
        infoList.append("clear")
        return infoList
    else:
        return infoList

def identifyImage(img):
    '''
    This function takes an image as input and identifies it as one of the environment objects using Keras.
    '''
    class_names = ['greybox', 'openroom', 'redsphere']

    target = (240, 240) #target size

    #preparation section
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img = img.resize(target)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = identifyModel.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    imageName = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return (str(imageName), confidence) #you can adjust the return value so it only classifies image of a certain confidence. We leave this out for initial testing


def processImageSentiment(environmentFeature, event):
    '''
    This function takes an image as input, identifies it, and obtains the sentiment based on the event that
    was analyzed either from the robot or from our JSON ruleset (see below). For instance, if a robot suffers damage or
    ceases to function, it will
    update the image sentiment to be higher. Damage to a robot may cause a higher sentiment, while loss of
    function will lead to the maximum sentiment, which is defined here as 3,
    though it can be tuned to any level.

    Robots can be tuned to avoid or approach objects based on sentiment gained from the environment.
    In this demonstration, a robot would avoid an object if its sentiment was 3 or higher,
    and it would report caution if its sentiment was 1 or higher to showcase the difference in response.

    Sentiment is stored in Redis and can also be stored permanently for future tasks, allowing
    information about environments to be gathered over time. In this server we reset the sentiment each time for demonstration.

    This function also reads and utilizes our JSON ruleset. 
    We use the JSON ruleset to deduce that if a robot loses contact and its last seen object is a (given item, but in the paper a red bowl)
    we should mark the item as dangerous. This ruleset can be modified at any time to produce new behavior as long as the relevant case
    is considered in the edge or robot code. 
    '''
    ruleset = {}
    with open("ruleset.json") as j:
        ruleset = json.load(j) #load ruleset to use for evaluation


    currentSentiment = 0
    maxSentiment = 3 #define max sentiment
    evaluation = evaluateEvent(event)
    if environmentFeature != "unrecognized" and environmentFeature != "openroom":
        if evaluation[0] == "DAMAGE_DETECTED":
            currentSentiment = int(r.get(environmentFeature))
            if currentSentiment > 3:
                pass
            else:
                r.mset({str(environmentFeature): str(currentSentiment+1)})
    print("evaluation is: " + str(evaluation))
    if evaluation[0] == "LOST_CONTACT":
        print("lost contact triggered")
        id = evaluation[1]
        key = str(id) + "lastseen" 
        lastSeen = r.get(key) #get last seen object
        lastSeen = lastSeen.decode()
        print("lastSeen is: " + str(lastSeen))
        action = ruleset["negativeRules"][evaluation[0]] #match LOST_CONTACT to desired outcome which is to avoid object
        print("action is: " + str(action))
        if action == "SET_MAX_SENTIMENT": #in this case, we should set the max sentiment to 3 in accordance with our ruleset. Other outcomes are possible if we modify this. 
            print("setting max sentiment of " + str(lastSeen) + "!")
            r.mset({str(lastSeen): str(maxSentiment)})

    newSentiment = int(r.get(environmentFeature))
    print("final new sentiment is: " + str(newSentiment))
    if newSentiment >= 3:
        return (environmentFeature, "avoid")
    elif newSentiment > 0:
        return (environmentFeature, "caution")
    else:
        return(environmentFeature, "clear")


@app.route('/receive', methods = ['POST'])
def receive():
    '''
    To monitor loss of contact, the receive function logs the time between each
    POST from active robots. If a predefined period lapses, the robot will be presumed  (or otherwise as marked in JSON ruleset, see above).
    Robots can also report damage themselves through the event.
    '''
    img = request.files['image'].read() #current environment image
    event = request.args.get('event')
    id = request.args.get('id')
    print("event is " + str(event))
    print("id is " + str(id))
    setContact(id, time.time()) #we've heard from this robot
    environmentFeature = identifyImage(img)[0] #use model on image
    print("environment image identified as: " + str(environmentFeature))
    key = str(id) + "lastseen"
    print("robot checking in: key is " + key + " and value is: " + str(environmentFeature))
    print("contact time is: " + str(getContact(id)))
    r.mset({key: environmentFeature}) #store robot last seen image in redis
    result = processImageSentiment(environmentFeature, event)
    print("result is: " + str(result))
    return str(result)

if __name__ == '__main__':
    '''
    Initialize the model and feature sentiment. This demonstration simply distinquishes between the 3D-printed
    houses (greybox), the red plastic bowl (redsphere), and the open environment (openroom or which is not a given environmental feature).
    Any number of features, of course, can be loaded, given that they exist in the model.
    '''
    r = redis.Redis() #initialize Redis connection
    identifyModel = tf.keras.models.load_model('../imagetraining/saved_modelz3/my_model')
    robotIds = ['1', '2', '3'] #initialize number and IDs of robots
    for environmentFeature in ['greybox', 'redsphere', 'openroom']: #default sentiment is 0 or neutral
        r.mset({environmentFeature: "0"})
    for rid in robotIds: #reset robot contact and image data
        key = str(rid) + "contact"
        r.delete(key)
        key = str(rid) + "lastseen"
        r.delete(key)

    app.run(host = '0.0.0.0', port = 5000)
