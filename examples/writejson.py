# -*- coding: utf-8 -*-
import json
import time
a=10
def store(data):
    with open('data.json', 'w') as json_file:
        json_file.write(json.dumps(data))

def load():
    with open('data.json') as json_file:
        data = json.load(json_file)
        return data



if __name__ == "__main__":

    data = {}
    data["last"]=time.strftime("%Y%m%d")
    data["framerate"]=("%d",a)
    store(data)

    data = load()
    #print data["last"]
