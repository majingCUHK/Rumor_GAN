# -*- coding: utf-8 -*-
"""
@task: save model, load model 
@author: majing
@time: Tue Nov 10 16:29:42 2015
"""
import numpy as np
import cPickle as pickle

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

############################ save model #######################################
## for GAN model ##
def save_model(f, model):
    ps = {}
    for p in model.params_gen:
        ps[p.name] = p.get_value() 
    for p in model.params_dis:
        ps[p.name] = p.get_value()   
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    print "Saved model parameters to %s." % f
    
def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params_gen:
        p.set_value(ps[p.name])
    for p in model.params_dis:
        p.set_value(ps[p.name])   
    print "loaded model parameters from %s." % f    
    return model    
    
## generator
def save_model_gen(f, model):
    ps = {}
    for p in model.params_gen:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    print "Saved generator parameters to %s." % f
    
def load_model_gen(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params_gen:
        p.set_value(ps[p.name])    
    print "loaded generator parameters from %s." % f    
    return model

## discriminator 
def save_model_dis(f, model):
    ps = {}
    for p in model.params_dis:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    print "Saved discriminator parameters to %s." % f
    
def load_model_dis(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params_dis:
        p.set_value(ps[p.name])
    print "loaded discriminator parameters from %s." % f    
    return model
    