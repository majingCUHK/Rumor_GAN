__doc__ = """GRU-RNN aka GAN. two seq2seq-based generators and one RNN-based discriminator."""

import numpy as np
import theano
from theano import tensor as T
from collections import OrderedDict
#from theano.compat.python2x import OrderedDict

theano.config.floatX = 'float64'


####### pre-defiend functions #######
def init_matrix(shape):
    return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)
            
def init_vector(shape):
    return np.zeros(shape, dtype=theano.config.floatX)
    
######################################    
class GAN():
    def __init__(self, vocab_size, hidden_size=5, Nclass=2, momentum=0.9):
        self.X_word = T.matrix('x_word') ## input X index
        #self.X_index = T.imatrix('x_index')
        self.Len = T.iscalar('Len') ## generator sentence length
        self.Y = T.vector('Y') # ground truth
        self.Yg = T.vector('Yg') # generated label for x
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.Nclass = Nclass
        self.momentum = momentum
        self.define_train_test_funcs()
        
    class Generator_NR():
        def __init__(self, vocab_size, hidden_size=5, Nclass=2, bptt_truncate=4):
            ##assign instance
            self.vocab_size = vocab_size ## vocabulary dim:5k 
            self.hidden_size = hidden_size ## hidden dim: 100
            self.Nclass = Nclass ## no. of classes:2 or 4
            self.bptt_truncate=bptt_truncate
            # emb layer & encoder
            self.Eg_en_nr = theano.shared(name='Eg_en_nr', value=init_matrix([self.hidden_size, self.vocab_size]))            
            self.Eg_de_nr = theano.shared(name='Eg_de_nr', value=init_matrix([self.vocab_size, self.hidden_size]))
            self.cg_de_nr = theano.shared(name='cg_de_nr', value=init_vector([self.vocab_size]))
            # GRU: encoder
            self.Wg_en_nr = theano.shared(name='Wg_en_nr', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.Ug_en_nr = theano.shared(name='Ug_en_nr', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.bg_en_nr = theano.shared(name='bg_en_nr', value=init_vector([3, self.hidden_size]))
            # GRU: decoder
            self.Wg_de_nr = theano.shared(name='Wg_de_nr', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.Ug_de_nr = theano.shared(name='Ug_de_nr', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.bg_de_nr = theano.shared(name='bg_de_nr', value=init_vector([3, self.hidden_size]))  
                        
            self.params_gnr = [self.Eg_en_nr, self.Eg_de_nr, self.cg_de_nr, self.Wg_en_nr,self.Ug_en_nr, self.bg_en_nr,
                               self.Wg_de_nr,self.Ug_de_nr, self.bg_de_nr]
            
        def generate(self, x, l): 
            ## stage1: encode sentence ##
            # x:matrix; l:int;            
            def encode_sen_step(xt, st_prev):
                xe = self.Eg_en_nr.dot(xt)
                zt = T.nnet.sigmoid(self.Wg_en_nr[0].dot(xe)+self.Ug_en_nr[0].dot(st_prev)+self.bg_en_nr[0])
                rt = T.nnet.sigmoid(self.Wg_en_nr[1].dot(xe)+self.Ug_en_nr[1].dot(st_prev)+self.bg_en_nr[1])
                ct = T.tanh(self.Wg_en_nr[2].dot(xe)+self.Ug_en_nr[2].dot(st_prev * rt)+self.bg_en_nr[2])
                st = zt*st_prev + (1-zt)*ct
                return st
            
            s_en_nr, updates = theano.scan(
                fn=encode_sen_step,
                sequences = x,
                outputs_info = dict(initial=T.zeros(self.hidden_size)))
            
            ## stage2: decode sentence ##
            #h1 = T.tanh(self.Vg.dot(s_en[-1]) + self.Vgy[:,y] + self.cg) ## size:100*1
            w1 = T.nnet.relu( self.Eg_de_nr.dot(s_en_nr[-1]) + self.cg_de_nr )
            
            def decode_step(wt_prev, st_prev):
                xe = self.Eg_en_nr.dot(wt_prev)
                zt = T.nnet.sigmoid(self.Wg_de_nr[0].dot(xe)+self.Ug_de_nr[0].dot(st_prev)+self.bg_de_nr[0])
                rt = T.nnet.sigmoid(self.Wg_de_nr[1].dot(xe)+self.Ug_de_nr[1].dot(st_prev)+self.bg_de_nr[1])
                ct = T.tanh(self.Wg_de_nr[2].dot(xe)+self.Ug_de_nr[2].dot(st_prev * rt)+self.bg_de_nr[2])
                st = zt*st_prev + (1-zt)*ct
    
                wt = T.nnet.relu( self.Eg_de_nr.dot(st) + self.cg_de_nr )
                return [wt, st]
            
            (words, _), updates = theano.scan(
                fn=decode_step,
                outputs_info = [w1, s_en_nr[-1]],
                n_steps=l)
            return T.concatenate([[w1],words[:-1]])

    class Generator_RN():
        def __init__(self, vocab_size, hidden_size=5, Nclass=2, bptt_truncate=4):
            ##assign instance
            self.vocab_size = vocab_size ## vocabulary dim:5k 
            self.hidden_size = hidden_size ## hidden dim: 100
            self.Nclass = Nclass ## no. of classes:2 or 4
            self.bptt_truncate=bptt_truncate
            # emb layer & encoder
            self.Eg_en_rn = theano.shared(name='Eg_en_rn', value=init_matrix([self.hidden_size, self.vocab_size]))            
            self.Eg_de_rn = theano.shared(name='Eg_de_rn', value=init_matrix([self.vocab_size, self.hidden_size]))
            self.cg_de_rn = theano.shared(name='cg_de_rn', value=init_vector([self.vocab_size])) 
            # GRU: encoder
            self.Wg_en_rn = theano.shared(name='Wg_en_rn', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.Ug_en_rn = theano.shared(name='Ug_en_rn', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.bg_en_rn = theano.shared(name='bg_en_rn', value=init_vector([3, self.hidden_size]))
            # GRU: decoder
            self.Wg_de_rn = theano.shared(name='Wg_de_rn', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.Ug_de_rn = theano.shared(name='Ug_de_rn', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.bg_de_rn = theano.shared(name='bg_de_rn', value=init_vector([3, self.hidden_size]))
                       
            self.params_grn = [self.Eg_en_rn, self.Eg_de_rn, self.cg_de_rn, self.Wg_en_rn, self.Ug_en_rn,self.bg_en_rn,
                               self.Wg_de_rn, self.Ug_de_rn, self.bg_de_rn]
            
        def generate(self, x, l): 
            ## stage1: encode sentence ##
            # x:matrix; l:int;            
            def encode_sen_step(xt, st_prev):
                xe = self.Eg_en_rn.dot(xt)
                zt = T.nnet.sigmoid(self.Wg_en_rn[0].dot(xe)+self.Ug_en_rn[0].dot(st_prev)+self.bg_en_rn[0])
                rt = T.nnet.sigmoid(self.Wg_en_rn[1].dot(xe)+self.Ug_en_rn[1].dot(st_prev)+self.bg_en_rn[1])
                ct = T.tanh(self.Wg_en_rn[2].dot(xe)+self.Ug_en_rn[2].dot(st_prev * rt)+self.bg_en_rn[2])
                st = zt*st_prev + (1-zt)*ct
                return st
            
            s_en_rn, updates = theano.scan(
                fn=encode_sen_step,
                sequences = x,
                outputs_info = dict(initial=T.zeros(self.hidden_size)))
            
            ## stage2: decode sentence ##
            #h1 = T.tanh(self.Vg.dot(s_en[-1]) + self.Vgy[:,y] + self.cg) ## size:100*1
            w1 = T.nnet.relu( self.Eg_de_rn.dot(s_en_rn[-1]) + self.cg_de_rn )            
            
            def decode_step(wt_prev, st_prev):
                xe = self.Eg_en_rn.dot(wt_prev)
                zt = T.nnet.sigmoid(self.Wg_de_rn[0].dot(xe)+self.Ug_de_rn[0].dot(st_prev)+self.bg_de_rn[0])
                rt = T.nnet.sigmoid(self.Wg_de_rn[1].dot(xe)+self.Ug_de_rn[1].dot(st_prev)+self.bg_de_rn[1])
                ct = T.tanh(self.Wg_de_rn[2].dot(xe)+self.Ug_de_rn[2].dot(st_prev * rt)+self.bg_de_rn[2])
                st = zt*st_prev + (1-zt)*ct
    
                wt = T.nnet.relu( self.Eg_de_rn.dot(st) + self.cg_de_rn )
                return [wt, st]
            
            (words, _), updates = theano.scan(
                fn=decode_step,
                outputs_info = [w1, s_en_rn[-1]],
                n_steps=l)
            return T.concatenate([[w1],words[:-1]])
            
    class Discriminator():
        def __init__(self, vocab_size, hidden_size=5, Nclass=2, bptt_truncate=4):
            ##assign instance
            self.vocab_size = vocab_size ## vocabulary dim:5k 
            self.hidden_size = hidden_size ## hidden dim: 100
            self.Nclass = Nclass ## no. of classes:2 or 4 
            self.bptt_truncate=bptt_truncate 
            self.Eg_en = theano.shared(name='Eg_en', value=init_matrix([self.hidden_size, self.vocab_size]))
            self.Wd = theano.shared(name='Wd', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.Ud = theano.shared(name='Ud', value=init_matrix([3, self.hidden_size, self.hidden_size]))
            self.bd = theano.shared(name='bd', value=init_vector([3, self.hidden_size]))
            self.Vd = theano.shared(name='Vd', value=init_matrix([self.Nclass, self.hidden_size]))
            self.cd = theano.shared(name='cd', value=init_vector([self.Nclass]))
            self.params_d = [self.Eg_en, self.Wd, self.Ud, self.bd, self.Vd,self.cd]
            
        def discriminate(self, x): 
            def _recurrence(xt, st_prev):
                xe = self.Eg_en.dot(xt)
                zt = T.nnet.hard_sigmoid(self.Wd[0].dot(xe)+self.Ud[0].dot(st_prev)+self.bd[0])
                rt = T.nnet.hard_sigmoid(self.Wd[1].dot(xe)+self.Ud[1].dot(st_prev)+self.bd[1])
                ct = T.tanh(self.Wd[2].dot(xe)+self.Ud[2].dot(st_prev * rt)+self.bd[2])
                st = zt*st_prev + (1-zt)*ct
                return st
            
            s_d, updates = theano.scan(
                fn=_recurrence,
                sequences = x,
                outputs_info = dict(initial=T.zeros(self.hidden_size)))
            
            avgS = s_d[-1]
            prediction_c = T.nnet.softmax( self.Vd.dot(avgS)+self.cd )
            return prediction_c
            
        def contCmp(self, Xw, Xe):
            results, updates = theano.scan(
                 lambda xw,xe: T.mean( (xw-xe)**2 ), 
                 sequences=[Xw, Xe])
            return T.mean(results)#, T.sum(results)
            
    def define_train_test_funcs(self):
          G_NR = self.Generator_NR(self.vocab_size, self.hidden_size, self.Nclass)
          G_RN = self.Generator_RN(self.vocab_size, self.hidden_size, self.Nclass)
          D = self.Discriminator(self.vocab_size, self.hidden_size, self.Nclass)
          
          self.params_dis = D.params_d
          self.params_gnr = G_NR.params_gnr
          self.params_grn = G_RN.params_grn
          self.params_gen = G_NR.params_gnr + G_RN.params_grn          
          lr = T.scalar("lr")
          
          ## step1: update generator NR
          X_nr = G_NR.generate(self.X_word, self.Len) # step 1.1: X_n -> G_nr -> X_nr
          self.gen_nr = theano.function(inputs = [self.X_word, self.Len], outputs = X_nr)
                                          
          dgnr = D.discriminate( X_nr )
          self.d_gen_nr = theano.function(inputs = [self.X_word, self.Len], outputs = dgnr)
                    
               
          X_nrn = G_RN.generate(X_nr, self.Len) # step 1.2: X_nr -> G_rn -> X_nrn
          self.gen_nrn = theano.function(inputs = [self.X_word, self.Len], outputs = X_nrn)
          
          loss_nrn = D.contCmp(self.X_word, X_nrn)  # loss_rec 
          self.f_loss_nrn = theano.function(inputs = [self.X_word, self.Len], outputs = loss_nrn)

          loss_gnr = T.sum( (dgnr-self.Yg)**2 ) # loss_D
          self.loss_gen_nr = theano.function(inputs = [self.X_word, self.Len, self.Yg], outputs = loss_gnr)
     
          #loss_nr = loss_gnr + 0.02*loss_nrn 
          loss_nr = loss_gnr + loss_nrn 
          self.f_loss_nr = theano.function(inputs = [self.X_word, self.Len, self.Yg], outputs = loss_nr)

          gparams_gnr_pre = [] ## only loss_D
          for param in self.params_gnr:
              gparam = T.grad(loss_gnr, param)
              gparams_gnr_pre.append(gparam)
          updates_gnr_pre = self.gradient_descent(self.params_gnr, gparams_gnr_pre, lr)
          self.train_gnr_pre = theano.function(inputs = [self.X_word, self.Len, self.Yg, lr], updates = updates_gnr_pre)
          
          gparams_gnr = [] ## loss_D +lossC
          for param in self.params_gen:
              gparam = T.grad(loss_nr, param)
              gparams_gnr.append(gparam)
          updates_gnr= self.gradient_descent(self.params_gen, gparams_gnr, lr)
          self.train_gnr = theano.function(inputs = [self.X_word, self.Len, self.Yg, lr], updates = updates_gnr)

          ## step2: update generator RN
          X_rn = G_RN.generate(self.X_word, self.Len) # step 2.1: X_r -> G_rn -> X_rn
          self.gen_rn = theano.function(inputs = [self.X_word, self.Len], outputs = X_rn)
                                          
          dgrn = D.discriminate( X_rn )
          self.d_gen_rn = theano.function(inputs = [self.X_word, self.Len], outputs = dgrn)
                    
          loss_grn = T.sum( (dgrn-self.Yg)**2 ) # loss_D
          self.loss_gen_rn = theano.function(inputs = [self.X_word, self.Len, self.Yg], outputs = loss_grn)
          
          X_rnr = G_NR.generate(X_rn, self.Len) # step 2.2: X_rn -> G_nr -> X_rnr
          self.gen_rnr = theano.function(inputs = [self.X_word, self.Len], outputs = X_rnr)
          
          loss_rnr = D.contCmp(self.X_word, X_rnr) # loss_rec
          self.f_loss_rnr = theano.function(inputs = [self.X_word, self.Len], outputs = loss_rnr)
          
          #loss_rn = loss_grn + 0.02*loss_rnr 
          loss_rn = loss_grn + loss_rnr 
          self.f_loss_rn = theano.function(inputs = [self.X_word, self.Len, self.Yg], outputs = loss_rn)
          
          gparams_grn = [] 
          for param in self.params_gen:
              gparam = T.grad(loss_rn, param)
              gparams_grn.append(gparam)
          updates_grn = self.gradient_descent(self.params_gen, gparams_grn, lr)
          self.train_grn = theano.function(inputs = [self.X_word, self.Len, self.Yg, lr], updates = updates_grn)
          
          gparams_grn_pre = [] ## only loss_D
          for param in self.params_grn:
              gparam = T.grad(loss_grn, param)
              gparams_grn_pre.append(gparam)
          updates_grn_pre = self.gradient_descent(self.params_grn, gparams_grn_pre, lr)
          self.train_grn_pre = theano.function(inputs = [self.X_word, self.Len, self.Yg, lr], updates = updates_grn_pre)
          
          ## step3: update discriminate     
          d1 = D.discriminate(self.X_word) # for orignal X
          self.dis1 = theano.function(inputs = [self.X_word], outputs = d1)
          loss_d1 = T.sum( (d1-self.Y)**2 )
          self.loss_dis1 = theano.function(inputs = [self.X_word, self.Y], outputs = loss_d1)
          gparams_d = []
          for param in self.params_dis:
              gparam = T.grad(loss_d1, param)
              gparams_d.append(gparam)
          updates_d = self.gradient_descent(self.params_dis, gparams_d, lr)
          self.train_d = theano.function(inputs = [self.X_word, self.Y, lr], updates = updates_d)
            
          loss_dgnr = T.sum( (dgnr-self.Yg)**2 ) # for X_nr
          self.loss_dis2 = theano.function(inputs = [self.X_word, self.Len, self.Yg], outputs = loss_dgnr)
          gparams_dnr = []
          for param in self.params_dis:
              gparam = T.grad(loss_dgnr, param)
              gparams_dnr.append(gparam)
          updates_dnr = self.gradient_descent(self.params_dis, gparams_dnr, lr)
          self.train_dnr = theano.function(inputs = [self.X_word, self.Len, self.Yg, lr], updates = updates_dnr)
          
          loss_dgrn = T.sum( (dgrn-self.Yg)**2 ) # for X_rn
          self.loss_dis3 = theano.function(inputs = [self.X_word, self.Len, self.Yg], outputs = loss_dgrn)
          gparams_drn = []
          for param in self.params_dis:
              gparam = T.grad(loss_dgrn, param)
              gparams_drn.append(gparam)
          updates_drn = self.gradient_descent(self.params_dis, gparams_drn, lr)
          self.train_drn = theano.function(inputs = [self.X_word, self.Len, self.Yg, lr], updates = updates_drn)

          loss_dgnrn = T.sum( (D.discriminate(X_nrn)-self.Y)**2 ) # for X_nrn
          self.loss_dis4 = theano.function(inputs = [self.X_word, self.Len, self.Y], outputs = loss_dgnrn)
          gparams_dnrn = []
          for param in self.params_dis:
              gparam = T.grad(loss_dgnrn, param)
              gparams_dnrn.append(gparam)
          updates_dnrn = self.gradient_descent(self.params_dis, gparams_dnrn, lr)
          self.train_dnrn = theano.function(inputs = [self.X_word, self.Len, self.Y, lr], updates = updates_dnrn)
          
          loss_dgrnr = T.sum( (D.discriminate(X_rnr)-self.Y)**2 ) # for X_rnr
          self.loss_dis5 = theano.function(inputs = [self.X_word, self.Len, self.Y], outputs = loss_dgrnr)
          gparams_drnr = []
          for param in self.params_dis:
              gparam = T.grad(loss_dgrnr, param)
              gparams_drnr.append(gparam)
          updates_drnr = self.gradient_descent(self.params_dis, gparams_drnr, lr)
          self.train_drnr = theano.function(inputs = [self.X_word, self.Len, self.Y, lr], updates = updates_drnr)
          
          gparams_dnr2 = []
          for param in self.params_dis:
              gparam = T.grad(0.3*loss_dgnr+0.5*loss_d1+0.2*loss_dgnrn, param)
              gparams_dnr2.append(gparam)
          updates_dnr2 = self.gradient_descent(self.params_dis, gparams_dnr2, lr)
          self.train_dnr2 = theano.function(inputs = [self.X_word, self.Len, self.Y, self.Yg, lr], updates = updates_dnr2)
          
          gparams_drn2 = []
          for param in self.params_dis:
              gparam = T.grad(0.3*loss_dgrn+0.5*loss_d1+0.2*loss_dgrnr, param)
              gparams_drn2.append(gparam)
          updates_drn2 = self.gradient_descent(self.params_dis, gparams_drn2, lr)
          self.train_drn2 = theano.function(inputs = [self.X_word, self.Len, self.Y, self.Yg, lr], updates = updates_drn2)

    def gradient_descent(self, params, gparams, learning_rate):
        """Momentum GD with gradient clipping."""
        #grad = T.grad(loss, self.params)
        self.momentum_velocity_ = [0.] * len(gparams)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(params, gparams)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates
        
    ##################### calculate total loss #######################    
    # only loss D    
    def calculate_total_loss_gen_nr(self, xw, l, yg):
        ## x: all instances
        L = 0
        ## for each training instance
        for i in range(len(yg)):
            L += self.loss_gen_nr(xw[i], l[i], yg[i])
        return L/len(yg) 
    
    def calculate_total_loss_gen_rn(self, xw, l, yg):
        ## x: all instances
        L = 0
        ## for each training instance
        for i in range(len(yg)):
            L += self.loss_gen_rn(xw[i], l[i], yg[i])
        return L/len(yg) 
    # only loss D + loss Cont    
    def calculate_total_loss_gen_cnr(self, xw, l, yg):
        ## x: all instances
        Lg, Lc, L = 0, 0, 0
        ## for each training instance
        for i in range(len(yg)):
            L += self.f_loss_nr(xw[i], l[i], yg[i])
            Lg += self.loss_gen_nr(xw[i], l[i], yg[i])
            Lc += self.f_loss_nrn(xw[i], l[i])
        return Lg/len(yg), Lc/len(yg), L/len(yg) 
    
    def calculate_total_loss_gen_crn(self, xw, l, yg):
        ## x: all instances
        Lg, Lc, L = 0, 0, 0
        ## for each training instance
        for i in range(len(yg)):
            L += self.f_loss_rn(xw[i], l[i], yg[i])
            Lg += self.loss_gen_rn(xw[i], l[i], yg[i])
            Lc += self.f_loss_rnr(xw[i], l[i])
        return Lg/len(yg), Lc/len(yg), L/len(yg) 
        
    def calculate_total_loss_dis(self, xw, y):
        ## x: all instances
        L = 0
        ## for each training instance
        for i in range(len(y)):
            L += self.loss_dis1(xw[i], y[i])
        return L/len(y) 
        
    def calculate_total_loss_dis_gen(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
        ## x: all instances
        ## y -> yg
        L1 = 0
        ## for each training instance: Xnr
        for i in range(len(y_t)):
            L1 += self.loss_dis2(xw_t[i], l_t[i], y_t[i])
        ## for each training instance: Xrn
        for i in range(len(y_f)):
            L1 += self.loss_dis3(xw_f[i], l_f[i], y_f[i])   
        return L1/(len(y_t)+len(y_f))
        
    def calculate_total_loss_dis_gen2(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
        ## x: all instances
        ## y -> yg
        L2 = 0
        #print len(y_t),len(xw_t), len()len(y_f)
        ## for each training instance: Xnr
        for i in range(len(y_t)):
            L2 += self.loss_dis4(xw_t[i], l_t[i], y_t[i])
        ## for each training instance: Xrn
        for i in range(len(y_f)):
            L2 += self.loss_dis5(xw_f[i], l_f[i], y_f[i])
        return L2/(len(y_t)+len(y_f))
        