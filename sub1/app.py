import os
import pickle
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter


SLACK_TOKEN = os.getenv('SLACK_TOKEN')
SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET')
# SLACK_TOKEN='xoxb-507382705603-731558944624-dmNLQVhLnrckmPcMFCBSTJlL'
# SLACK_SIGNING_SECRET='00f81fdc04ebbe2281aade6066218bd6'

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req. 2-1-1. pickle로 저장된 model.clf 파일 불러오기

with open('model.clf', 'rb') as f:
    lrmodel = pickle.load(f)


beta_0 = lrmodel.coef_[0]
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.intercept_

def expected_sales(tv, rd, newspaper, beta_0, beta_1, beta_2, beta_3):
   return (tv*beta_0 + rd*beta_1 + newspaper*beta_2 + beta_3)

# Req. 2-1-2. 입력 받은 광고비 데이터에 따른 예상 판매량을 출력하는 lin_pred() 함수 구현
def lin_pred(test_str):
    L = test_str.split()
    if len(L) != 4:
        return '광고비를 tv, radio, newspaper 순으로 띄어서 입력해주세요'
    else:
        return "예상 판매량 : {:.5f}".format(expected_sales(float(L[1]), float(L[2]),float(L[3]), beta_0, beta_1, beta_2, beta_3))

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]

    keywords = lin_pred(text)
    slack_web_client.chat_postMessage(
        channel=channel,
        text=keywords
    )

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()

# if __name__ == '__main__':
#     app.run(host=os.getenv('IP'), port=os.getenv('PORT'), debug=True)