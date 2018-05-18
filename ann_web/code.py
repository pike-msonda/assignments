import os
import web
import pickle
from lib.ann import ANN
from lib.utils import *

urls =(
    '/', 'index',
    '/upload', "upload"
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
        x = web.input(myfile={})
        web.debug(x)
        web.debug(x['myfile'].filename) # This is the filename
        web.debug(x['myfile'].value) # This is the file contents
        web.debug(x['myfile'].file.read()) # Or use a file(-like) object
        raise web.seeother('/data')

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()