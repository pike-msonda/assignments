import os
import web
from lib.ann import ANN
from lib.utils import *
import pickle

urls = (
    '/', 'index'
)

# Get templates to render
render =  web.template.render('templates/')

x_train, x_test, y_train, y_test, dimension, classes =  prepare_data()
ann = ANN(x_train,x_test,y_train,y_test, dimension, classes,MODELFILENAME)
model = ann.train()

class index:
    def GET(self):
        """

        """
        return render.index()

class operation:
    def GET(self):
        """
            Insert cool AI stuff here
        """
        return "Will return something cool"
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()