import os
import web
import json
import pickle
import pandas as pd
from multiprocessing import Process, Queue
from lib.ann import ANN
from lib.utils import *

urls =(
    '/', 'index',
    '/upload', "upload",
    '/csvhanlder', "csvhanlder",
    '/neuron', "neuron"
)
# Get templates to render
render =  web.template.render("templates/")

class index:
    def GET(self):
        return render.index()
    
    def POST(self):
        data = web.input()
        web.debug(data)
        if not "new_model" in data:
            data.__setattr__("new_model", "off")

        ann = ANN(filename =  MODELFILENAME,
                  epochs = int(data.epoch),
                  learning = float(data.learning), 
                  hidden = int(data.hidden), 
                  decay_rate = float(data.decay_rate), 
                  new_model = data.new_model)
        #TODO: Make a thread safe execution
        #model = ann.train()# pass necessary tune-up variables here
        model = process_start(target=trainer, args=[ann])
        accuracy, train_error, test_error = ann.accuracy(model)
        image = process_start(target=graphpainter, args=[ann,model])
        response = json.dumps({'Acc':accuracy,'TrainE':train_error,'TestE':test_error,'Process':model.process, "Figure":image }, 
        sort_keys=True, indent=2, separators=(',',':'))
        return response
        
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