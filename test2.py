import pickle
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter


with open('model.clf', 'rb') as f:
    data = pickle.load(f)


print(data)