# senior-research
Code for my senior research project: "Image Classification with Knowledge-Based Systems on the Edge for Real-Time Danger Avoidance in Robots".

If you want to run this project yourself, you'll need a raspberry-pi powered robot (some construction details are in the paper) and a server with Flask, Redis, and Tensorflow installed. To use it with your own trained model, you can adjust the setup in servertraining.py and then mirror those trained classes and their outcomes in the edgeServer and ruleset files. This will allow you to define custom rulesets that correspond to some event in the edge server. 
