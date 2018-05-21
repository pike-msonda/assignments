import os
import web
import json
import pickle
import pandas as pd
from lib.ann import ANN
from lib.utils import *


urls =(
    '/', 'index',
    '/upload', "upload",
    '/csvhanlder', "csvhanlder"
)
# Get templates to render
render =  web.template.render("templates/")

class index:
    def GET(self):
        return render.index()

    def POST(self):
        """
            Cool stuff will go here

        """
        data = web.data()
        web.debug(data)
        x_train, x_test, y_train, y_test, dimensions, classes = prepare_data()

        ann = ANN(x_train, x_test, y_train, y_test, dimensions,classes, MODELFILENAME)
        #TODO: add tune-up variables to the "train" method in ANN class
        model = ann.train() # pass necessary tune-up variables here
        accuracy, train_error, test_error = ann.accuracy(model)

        return accuracy, train_error, test_error

class upload:
    def GET(self):
        return "hello world"

    def POST(self):
        x = web.input()
        #TODO: Add file writing process to a separate thread
        filename="data/"+x.name
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write(x.data)
        return  "Status Ok"

class csvhanlder:
    def GET(self):
        data =  read_data()
        return data.head(10).to_html()

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()