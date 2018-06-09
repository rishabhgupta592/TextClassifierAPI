#!/usr/bin/env python
"""
    File name: 
    Description:
    Author: Rishabh Gupta
    Date created:
    Date last modified:
    Python Version: 2.7
"""

#!flask/bin/python
from flask import Flask
from flask import request
import json
from nn import nn_runner as tcr

app = Flask(__name__)

@app.route('/intent_classify')
def intent_classify():
    user_input = request.args
    # data = request.data
    # user_input = json.loads(data)
    # print(request.data)
    # print(user_input)
    query = user_input.get('query')
    res = tcr.get_class(query)
    # print(res[0][0])
    intent_name = "undefined"
    intent = res[0]
    # print(intent[0])
    # print(intent[1])
    if intent[0] == 0.0:
        intent_name = "Policy 1"
    elif intent[0] == 1.0:
        intent_name = "Policy 2"
    #
    # print(res)
    jsondata = {'Query': query, 'Class': intent_name}
    return json.dumps(jsondata)

if __name__ == '__main__':
    app.run()