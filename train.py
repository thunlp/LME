# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
from math import ceil, exp, log
from sklearn.externals import joblib
from src.model.nn_model import Model
from src.batcher import Batcher
from src.hook import acc_hook, save_predictions
from src.model.create_prior_knowledge import create_LE

parser = argparse.ArgumentParser()
parser.add_argument("dataset",help="dataset to train model",choices=["figer","gillick"])
parser.add_argument("encoder",help="context encoder to use in model",choices=["averaging","lstm","attentive"])
parser.add_argument('--feature', dest='feature', action='store_true')
parser.add_argument('--no-feature', dest='feature', action='store_false')
parser.set_defaults(feature=False)
parser.add_argument('--hier', dest='hier', action='store_true')
parser.add_argument('--no-hier', dest='hier', action='store_false')
parser.set_defaults(hier=False)
parser.add_argument('--pre', dest='pre', action='store_true')
parser.add_argument('--id', dest='id', action='store')
parser.add_argument('--lamb', dest='lamb', action='store')
parser.add_argument('--hard', dest='hard', action='store_true') #gumbel
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--cs', dest='cs', action='store_true') #case study
args = parser.parse_args()

print "Loading the dictionaries"
d = "Wiki" if args.dataset == "figer" else "OntoNotes"
dicts = joblib.load("data/"+d+"/dicts_"+args.dataset+".pkl")
vocab_size = 20000
print("vocab_size:\t%d" %vocab_size)


print "Creating the model"
LE = create_LE("resource/"+d+"/label2id_"+args.dataset+".txt", dicts, vocab_size)
model = Model(type=args.dataset,encoder=args.encoder,
    hier=args.hier,feature=args.feature,LE=LE,vocab_size=vocab_size,
    lamb=float(args.lamb),LMout=dicts["id2vec"],hard=args.hard)

batch_size = 1000 if args.dataset=="figer" else 128

print "Loading the datasets"
if not args.test:
    train_dataset = joblib.load("data/"+d+"/train_"+args.dataset+".pkl")
    dev_dataset = joblib.load("data/"+d+"/dev_"+args.dataset+".pkl")
    print "train_size:", train_dataset["data"].shape[0]
    print "dev_size: ", dev_dataset["data"].shape[0]
    train_batcher = Batcher(train_dataset["storage"],train_dataset["data"],
        batch_size,10,dicts["id2vec"], vocab_size)
    dev_batcher = Batcher(dev_dataset["storage"],dev_dataset["data"],
        dev_dataset["data"].shape[0],10,dicts["id2vec"], vocab_size)


test_dataset = joblib.load("data/"+d+"/test_"+args.dataset+".pkl")
test_batch_size = test_dataset["data"].shape[0]
if args.cs:
    test_batch_size = 1
print "test_size: ", test_dataset["data"].shape[0]

test_batcher = Batcher(test_dataset["storage"],test_dataset["data"],
    test_batch_size,10,dicts["id2vec"], vocab_size)
if args.test:
    #only works for Wiki
    model.load_all("./Models/"+d+"/lamb"+str(args.lamb)+"/model")
    batch_data = test_batcher.next()
    scores = model.predict(batch_data)
    acc_hook(scores, batch_data["Y"])
    sys.exit(0)

step_par_epoch = train_dataset["data"].shape[0]/batch_size

if args.cs:
    #only works for wiki

    id2word = lambda y: map(lambda x: dicts["id2word"][x], y) #list of id to word
    sent = []
    enti = []
    ybase = []
    yseds = []
    y_ = []


    model.load_all()
    for i, line in enumerate(test_dataset["data"]):
        sent.append(id2word(test_dataset["storage"][line[0]:line[1]]))
        enti.append(id2word(test_dataset["storage"][line[2]:line[3]]))

        batch_data = test_batcher.next()
        y_.append(batch_data["Y"])
        yseds.append(model.printer([model.distribution], batch_data, True))

    model.load_all("./Emodel/baseline lamb0.0/keep/588-778-744/model-0")
    for i, line in enumerate(test_dataset["data"]):
        batch_data = test_batcher.next()
        ybase.append(model.printer([model.distribution], batch_data, True))

    fout = open("case.csv", 'w')

    for i in range(len(sent)):
        fout.write('"'+str(i)+'"\n')
        fout.write('"'+' '.join(sent[i])+'"\n')
        fout.write('"'+' '.join(enti[i])+'"\n')
        for j in range(113):
            if (y_[i][0][j]==1) or (ybase[i][0][0][j]>0.05) or (yseds[i][0][0][j]>0.05):
                tmp = '" ","' + dicts["id2label"][j] + '",' \
                    + '"' + str(ybase[i][0][0][j]) +'",' \
                    + '"' + str(yseds[i][0][0][j]) +'",'
                if y_[i][0][j]==1:
                    tmp += '"True"\n'
                elif (ybase[i][0][0][j]>0.5) or (yseds[i][0][0][j]>0.5):
                    tmp += '"Note"\n'
                else:
                    tmp += '""\n'
                fout.write(tmp)
        fout.write('"========================="\n')
    fout.close()

    sys.exit(0)


if args.pre:
    # pretrain language model
    if args.dataset=="gillick":
        model.load_LM("./Wiki/LMmodel/debug-1000/model")
    print("start pretraining language model")
    best_test_loss = 100
    for epoch in range(50):

        print "Test loss:\t",
        test_loss = []
        for i in range(int(ceil(float(
            test_dataset["data"].shape[0])/test_batch_size))):
            
            batch_data = test_batcher.next()
            a = model.printer([model.pre_loss], batch_data, mute=True)
            test_loss.append(a)
        test_loss = np.mean(test_loss)
        print(test_loss)

        if test_loss<best_test_loss:
            model.save_LM(epoch, args.id)
            best_test_loss = test_loss



        print('\n'+'='*90)
        print "epoch",epoch
        for i in range(step_par_epoch):
            if i%10==0:
                print "\r%d"%i,
            batch_data = train_batcher.next()

            if i%(step_par_epoch/5)==0:
                print "Train loss:\t",
                model.printer([model.pre_loss], batch_data)
                
            model.pretrain(batch_data)
            

        #end of one epoch


    sys.exit(0)


print "start trainning"
model.load_LM()
print('#'*100)
if args.id is None:
    save_id = "lamb"+str(model.lamb)
else:
    save_id = args.id+" lamb"+str(model.lamb)
print(save_id)
print('#'*100)

for epoch in range(10):
    # train_batcher.shuffle()
    print "epoch",epoch
    for i in range(step_par_epoch):
        print "\r%d"%i,
        batch_data = train_batcher.next()
        # if i%(step_par_epoch/10)==0:
        #     loss = model.printer([model.LM_loss_total, model.type_loss], batch_data)
        #     print(loss)
        model.train(batch_data)


    print "------dev--------"
    batch_data = dev_batcher.next()
    scores = model.predict(batch_data)
    acc_hook(scores, batch_data["Y"])
    
    
    model.load_all("./Models/"+d+"/lamb"+str(args.lamb)+"/model")
    print "-----test--------"
    batch_data = test_batcher.next()
    scores = model.predict(batch_data)
    acc_hook(scores, batch_data["Y"])
    print

#    model.save_all(epoch, save_id)

print "Training completed.  Below are the final test scores: "
print "-----test--------"
batch_data = test_batcher.next()
scores = model.predict(batch_data)
acc_hook(scores, batch_data["Y"])
fname = args.dataset + "_" + args.encoder + "_" + str(args.feature) + "_" + str(args.hier) + ".txt"
save_predictions(scores, batch_data["Y"], dicts["id2label"],fname)

print "Cheers!"