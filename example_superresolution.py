#%% Code to reproduce results in figures 16-19
import odl
import os
import src.application as application
import src.misc as misc
import src.models as models

gtruth, sinfo, operator, data = application.superresolution()

filename = 'superresolution'

folder_base = 'pics'
folder_out = '{}/{}'.format(folder_base, filename)

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

cases = [1, 2]

for case in cases:
    if case == 1:
        
        regparams = {
                'tv': [5e-4],
                'wtv': [5e-4],
                'dtv': [5e-4],
                'h1': [1e-5],
                'wh1': [1e-5],
                'dh1': [5e-5],
                'tgv': [5e-4],
                'wtgv': [10e-4],
                'dtgv': [10e-4],
                'jtv': [10e-4],
                'tnv': [10e-4]}
        
        regularizers = ['tv', 'h1', 'tgv', 'wtv', 'dtv', 'jtv', 'tnv', 'wh1', 'dh1', 
                        'wtgv', 'dtgv']
                
        etas_w = [5e-2]
        etas_d = [5e-2]
        etas_jtv = [20]
        etas_tnv = [10]
  
    elif case == 2:
        
        regparams = {
                'tv': [5e-4],
                'wtv': [5e-4],
                'dtv': [5e-4],
                'h1': [1e-5],
                'wh1': [1e-5],
                'dh1': [5e-5],
                'tgv': [5e-4],
                'wtgv': [10e-4],
                'dtgv': [10e-4],
                'jtv': [10e-4],
                'tnv': [10e-4]}
        
        regularizers = ['wh1', 'wtv', 'wtgv', 'dh1', 'dtv', 'dtgv', 'jtv', 'tnv']
         
        etas_w = [5e-1, 5e-3]   
        etas_d = [5e-1, 5e-3]
        etas_jtv = [.2, 2000]
        etas_tnv = [.1, 1000]
        
    else:
        raise ValueError('case not defined')

    betas_tgv = [0.05]        
    gammas_d = [0.9]
    
    betas = {'tv': [None],
            'h1': [None],
            'tgv': betas_tgv,
            'wtv': [None], 
            'dtv': [None],
            'jtv': [None], 
            'tnv': [None],
            'wh1': [None], 
            'dh1': [None],
            'wtgv': betas_tgv,
            'dtgv': betas_tgv}
    
    etas = {'tv': [None],
            'h1': [None],
            'tgv': [None],
            'wtv': etas_w,
            'dtv': etas_d,
            'jtv': etas_jtv, 
            'tnv': etas_tnv, 
            'wh1': etas_w, 
            'dh1': etas_d,
            'wtgv': etas_w,
            'dtgv': etas_d}
    
    gammas = {'tv': [None],
              'h1': [None],
              'tgv': [None],
              'wtv': [None],
              'dtv': gammas_d,
              'jtv': [None], 
              'tnv': [None], 
              'wh1': [None], 
              'dh1': gammas_d,
              'wtgv': [None],
              'dtgv': gammas_d}
    
    niter = 3000 + 1
    step = 500
    iter_save = []
    iter_plot = [niter-1]
    
    for regularizer in regularizers:
        for regparam in regparams[regularizer]:
            for eta in etas[regularizer]:
                for beta in betas[regularizer]:
                    for gamma in gammas[regularizer]:
                
                        prefix = '{}/{}_{}_alpha{:.1e}'.format(
                            folder_out, filename, regularizer, regparam)
                        
                        if eta is not None:
                            prefix += '_eta{:.1e}'.format(eta)
            
                        if beta is not None:
                            prefix += '_beta{:.1e}'.format(beta)
        
                        if gamma is not None:
                            prefix += '_gamma{:.1e}'.format(gamma)
                        
                        print(prefix)
    
                        if regularizer == 'h1':
                            grad = models.gradient(gtruth.space)
                            G, F, A = models.h1(operator, data, regparam, 
                                                grad=grad, datafit='l2')
                            
                        elif regularizer == 'tv':
                            grad = models.gradient(gtruth.space)
                            G, F, A = models.tv(operator, data, regparam, 
                                                grad=grad, datafit='l2')
                            
                        elif regularizer == 'wh1':
                            grad = models.gradient(gtruth.space, sinfo=sinfo, 
                                                   mode='location', eta=eta)
                            G, F, A = models.h1(operator, data, regparam, 
                                                grad=grad, datafit='l2')
                
                        elif regularizer == 'wtv':
                            grad = models.gradient(gtruth.space, sinfo=sinfo, 
                                                   mode='location', eta=eta)
                            G, F, A = models.tv(operator, data, regparam, 
                                                grad=grad, datafit='l2')
                            
                        elif regularizer == 'dh1':
                            grad = models.gradient(gtruth.space, sinfo=sinfo, 
                                                   mode='direction', gamma=gamma, 
                                                   eta=eta)
                            G, F, A = models.h1(operator, data, regparam, 
                                                grad=grad, datafit='l2')
                            
                        elif regularizer == 'dtv':
                            grad = models.gradient(gtruth.space, sinfo=sinfo, 
                                                   mode='direction', gamma=gamma, 
                                                   eta=eta)
                            G, F, A = models.tv(operator, data, regparam, 
                                                grad=grad, datafit='l2')
                
                        elif regularizer == 'jtv':
                            G, F, A = models.jtv(operator, data, regparam, sinfo,
                                                 eta, datafit='l2')
                            
                        elif regularizer == 'tnv':
                            G, F, A = models.tnv(operator, data, regparam, sinfo,
                                                 eta, datafit='l2')
                            
                        elif regularizer == 'tgv':
                            grad = models.gradient(gtruth.space)
                            G, F, A = models.tgv(operator, data, regparam, beta, 
                                                 grad=grad, datafit='l2')
                
                        elif regularizer == 'wtgv':
                            grad = models.gradient(gtruth.space, sinfo=sinfo,
                                                   mode='location', eta=eta)
                            G, F, A = models.tgv(operator, data, regparam, beta, 
                                                 grad=grad, datafit='l2')
                
                        elif regularizer == 'dtgv':
                            grad = models.gradient(gtruth.space, sinfo=sinfo, 
                                                   mode='direction', gamma=gamma, 
                                                   eta=eta)
                            G, F, A = models.tgv(operator, data, regparam, beta, 
                                                 grad=grad, datafit='l2')
                            
                        else:
                            D = None            
                                    
                        norm_As = []
                        for Ai in A:
                            xs = odl.phantom.white_noise(Ai.domain, seed=1807)
                            norm_As.append(Ai.norm(estimate=True, xstart=xs))
                            
                        Atilde = odl.BroadcastOperator(
                                *[Ai / norm_Ai for Ai, norm_Ai in zip(A, norm_As)])
                        Ftilde = odl.solvers.SeparableSum(
                                *[Fi * norm_Ai for Fi, norm_Ai in zip(F, norm_As)])
                        
                        obj_fun = Ftilde * Atilde + G
                        
                        Atilde_norm = Atilde.norm(estimate=True)
                
                        x = Atilde.domain.zero()
                        scaling = 1
                        sigma = scaling / Atilde_norm
                        tau = 0.999 / (scaling * Atilde_norm)

                        cb = (odl.solvers.CallbackPrintIteration(step=step, end=', ') &
                              odl.solvers.CallbackPrintTiming(step=step, cumulative=False, end=', ') &
                              odl.solvers.CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                                              cumulative=True) &
                              misc.MyCallback(iter_save, iter_plot, prefix, 
                                              obj_fun, gtruth, error=False))
                        
                        odl.solvers.pdhg(x, G, Ftilde, Atilde, niter, tau, sigma, callback=cb)