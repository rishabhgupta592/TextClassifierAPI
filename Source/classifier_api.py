#encoding=utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 12:08:33 2017

@author: Rishabh.gupta
"""
# URL signature
# http://localhost:8080/api_v1?userId=10&query=hi

import web, json
# import text_classifier_runner as tcr
from nn import nn_runner as tcr


urls = (

    '/intent_classify', 'Bot'
)

app = web.application(urls, globals())

class Bot:
    def GET(self):
        web.header('Content-Type', 'application/json')
        user_input = web.input()
        query = user_input.query
        res = tcr.get_class(query)
        print(res[0][0])
        intent = res[0][0]
        if intent == 0:
            intent_name = "a"
        elif intent == 1:
            intent_name ="401(k) Plan"

        jsondata = {'Query': query, 'Class': intent_name}
        return json.dumps(jsondata)


if __name__ == "__main__":
    app.run()
