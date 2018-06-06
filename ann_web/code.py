import os
import web
import json
import pickle
import pandas as pd
from multiprocessing import Process, Queue
from lib.ann import ANN
from lib.utils import *

"""
    Server Application. 
    Follow webpy standards. 
"""

#URL list
urls =(
    '/', 'index',
    '/upload', "upload",
    '/csvhanlder', "csvhanlder",
    '/neuron', "neuron",
    '/stats', "stats"
)
# Get templates to render
render =  web.template.render("templates/")

#Initialise application
app = web.application(urls, globals())

#Set global variables in here
def add_global_hook():
    upload_filename = ""
    data = ""
    g = web.storage({"filename": upload_filename, "data":data })

    def _wrapper(handler):
        web.ctx.globals = g
        return handler()
    return _wrapper


if web.config.get('_session') is None:
    session =  web.session.Session(app, web.session.DiskStore('sessions'))
    web.config._session = session
else:
    session = web.config._session


class index:
    def GET(self):
        return render.index()

    def POST(self):
        data = web.input()
        web.debug(data)
        if not "new_model" in data:
            data.__setattr__("new_model", "off")

        ann = ANN(filename =  web.ctx.globals.filename,
                  epochs = int(data.epoch),
                  learning = float(data.learning), 
                  hidden = int(data.hidden), 
                  decay_rate = float(data.decay_rate), 
                  new_model = data.new_model,
                  activation=data.activation,
                  momentum = float(data.momentum))

        model = process_start(target=trainer, args=[ann])
        accuracy, train_error, test_error = ann.accuracy(model)
        image = process_start(target=graphpainter, args=[ann,model])
        response = json.dumps({'Acc':accuracy,
                               'TrainE':train_error,
                               'TestE':test_error,
                               'Process':model.process, 
                               "Figure":image },
                               sort_keys=True, 
                               indent=2, 
                               separators=(',',':'))
        return response

class upload:
    def GET(self):
        return ""

    def POST(self):
        x = web.input()
        #TODO: Add file writing process to a separate thread
        filename="data/"+x.name
        web.ctx.globals.filename = filename
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write(x.data)
        return  "Uploaded."

class csvhanlder:
    def GET(self):
        web.ctx.globals.data =  read_data(web.ctx.globals.filename)
        data = web.ctx.globals.data
        return data.head(10).to_html()

class stats:
    def GET(self):
        inputs, outputs = const_values(web.ctx.globals.data)
        response = json.dumps({
            'inputs': inputs, 
            'outputs': outputs
        })
        return response


if __name__ == "__main__":
    app.add_processor(add_global_hook())
    app.run()
