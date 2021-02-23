#%% Models used in the examples
import odl
import numpy as np
import src.misc as misc

def gradient(space, sinfo=None, mode=None, gamma=1, eta=1e-2,
             show_sinfo=False, prefix=None):
    
    grad = odl.Gradient(space, method='forward', pad_mode='symmetric')
    
    if sinfo is not None:
        if mode == 'direction':
            norm = odl.PointwiseNorm(grad.range)
            grad_sinfo = grad(sinfo)
            ngrad_sinfo = norm(grad_sinfo)
              
            for i in range(len(grad_sinfo)):
                grad_sinfo[i] /= ngrad_sinfo.ufuncs.max()

            ngrad_sinfo = norm(grad_sinfo)            
            ngrad_sinfo_eta = np.sqrt(ngrad_sinfo ** 2 + eta ** 2)

            xi = grad.range.element([g / ngrad_sinfo_eta for g in grad_sinfo]) # UGLY
            
            Id = odl.operator.IdentityOperator(grad.range)
            xiT = odl.PointwiseInner(grad.range, xi)
            xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])
            
            grad = (Id - gamma * xixiT) * grad

            if show_sinfo:        
                misc.save_image(ngrad_sinfo, prefix + '_sinfo_norm')
                misc.save_vfield(xi.asarray(), filename=prefix + '_sinfo_xi')
                misc.save_vfield_cmap(filename=prefix + '_sinfo_xi_cmap')
            
        elif mode == 'location':
            norm = odl.PointwiseNorm(grad.range)
            ngrad_sinfo = norm(grad(sinfo))
            ngrad_sinfo /= ngrad_sinfo.ufuncs.max()

            w = eta / np.sqrt(ngrad_sinfo ** 2 + eta ** 2)
            grad = odl.DiagonalOperator(odl.MultiplyOperator(w), 2) * grad

            if show_sinfo:        
                misc.save_image(ngrad_sinfo, prefix + '_sinfo_norm')
                misc.save_image(w, prefix + '_w')
            
        else:
            grad = None

    return grad


def symm_derivative(space):
    Dx = odl.PartialDerivative(space, 0, method='backward', pad_mode='symmetric')
    Dy = odl.PartialDerivative(space, 1, method='backward', pad_mode='symmetric')

    # Create symmetrized operator and weighted space.
    # TODO: As the weighted space is currently not supported in ODL we find a
    # workaround.
    # W = odl.ProductSpace(U, 3, weighting=[1, 1, 2])
    # sym_gradient = odl.operator.ProductSpaceOperator(
    #    [[Dx, 0], [0, Dy], [0.5*Dy, 0.5*Dx]], range=W)
    return odl.operator.ProductSpaceOperator(
        [[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
  

def get_data_fit(datafit, data):
    if datafit == 'l1':         
        fun_datafit = odl.solvers.L1Norm(data.space).translated(data)
    elif datafit == 'l2':
        fun_datafit = odl.solvers.L2NormSquared(data.space).translated(data)
    else:
        fun_datafit = None
        
    return fun_datafit

    
def h1(operator, data, alpha, grad=None, nonneg=True, datafit=None):

    space = operator.domain
    
    if grad is None:
        grad = gradient(space)
        
    A = odl.BroadcastOperator(operator, grad)
    
    F1 = get_data_fit(datafit, data)  
    F2 = alpha * odl.solvers.L2NormSquared(grad.range)
    F = odl.solvers.SeparableSum(F1, F2)

    if nonneg:
        G = odl.solvers.IndicatorNonnegativity(space)
    else:
        G = odl.solvers.ZeroFunctional(space)
                    
    return G, F, A


def tv(operator, data, alpha, grad=None, nonneg=True, datafit=None):

    space = operator.domain
    
    if grad is None:
        grad = gradient(space)
    
    A = odl.BroadcastOperator(operator, grad)
    
    F1 = get_data_fit(datafit, data)
    F2 = alpha * odl.solvers.GroupL1Norm(grad.range)
    F = odl.solvers.SeparableSum(F1, F2)
    
    if nonneg:
        G = odl.solvers.IndicatorNonnegativity(space)
    else:
        G = odl.solvers.ZeroFunctional(space)        
            
    return G, F, A

    
def tgv(operator, data, alpha, beta, grad=None, nonneg=True, datafit=None):

    space = operator.domain
    
    if grad is None:
        grad = gradient(space)
        
    E = symm_derivative(space)
    I = odl.IdentityOperator(grad.range)
    
    A1 = odl.ReductionOperator(operator, odl.ZeroOperator(grad.range, operator.range))
    A2 = odl.ReductionOperator(grad, -I)
    A3 = odl.ReductionOperator(odl.ZeroOperator(space, E.range), E)
    A = odl.BroadcastOperator(*[A1, A2, A3])              
                  
    F1 = get_data_fit(datafit, data)
    F2 = alpha * odl.solvers.GroupL1Norm(grad.range)
    F3 = alpha * beta * odl.solvers.GroupL1Norm(E.range)
    F = odl.solvers.SeparableSum(F1, F2, F3)

    if nonneg:
        G = odl.solvers.SeparableSum(odl.solvers.ZeroFunctional(space),
                                     odl.solvers.ZeroFunctional(E.domain))

    else:    
        G = odl.solvers.SeparableSum(odl.solvers.IndicatorNonnegativity(space),
                                     odl.solvers.ZeroFunctional(E.domain))
            
    return G, F, A


def tnv(operator, data, alpha, sinfo, eta, nonneg=True, datafit=None):

    space = operator.domain
    grad = odl.Gradient(space) 
    
    P = odl.ComponentProjection(grad.range ** 2, 0)
    D = P.adjoint * grad
    Q = odl.ComponentProjection(grad.range ** 2, 1)
    A = odl.BroadcastOperator(operator, D)
    
    F1 = get_data_fit(datafit, data)
    N = odl.solvers.NuclearNorm(D.range, outer_exp=1, singular_vector_exp=1)
    F2 = alpha * N.translated(-Q.adjoint(eta * grad(sinfo)))
    F = odl.solvers.SeparableSum(F1, F2)
        
    if nonneg:
        G = odl.solvers.IndicatorNonnegativity(space)
    else:
        G = odl.solvers.ZeroFunctional(space)
                    
    return G, F, A


def jtv(operator, data, alpha, sinfo, eta, nonneg=True, datafit=None):

    space = operator.domain

    Dx = odl.PartialDerivative(space, 0)
    Dy = odl.PartialDerivative(space, 1)
    Z = odl.ZeroOperator(space)
    D = odl.BroadcastOperator(Dx, Dy, Z, Z)                                       
    A = odl.BroadcastOperator(operator, D)
    
    F1 = get_data_fit(datafit, data)
    Q = odl.BroadcastOperator(Z, Z, Dx, Dy)                    
    N = odl.solvers.GroupL1Norm(D.range)
    F2 = alpha * N.translated(-eta * Q(sinfo))
    F = odl.solvers.SeparableSum(F1, F2)
        
    if nonneg:
        G = odl.solvers.IndicatorNonnegativity(space)
    else:
        G = odl.solvers.ZeroFunctional(space)
        
    return G, F, A