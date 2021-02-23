#%% Additional functionality used for the applications
import numpy as np
import odl
from scipy.ndimage import rotate
from skimage.transform import resize
from skimage.measure import block_reduce

def get_resolution_pattern(space, smallest_rad, center=[0, 0], val=1, levels=4, 
                           rad_increase=1.2, pts_increase_add=None, 
                           pts_increase_mult=None, pts_level=3,
                           plot_center=True):
    dist = 2*smallest_rad
    
    o = center
    dist_origin = 0
    rad = smallest_rad
    
    ellipses = []
    
    if not isinstance(val, list):
        val = [val, val]
        
    if plot_center:
        #'value', 'axis_1', 'axis_2', 'center_x', 'center_y', 'rotation'
        ellipses += [[val[0], rad, rad, o[0], o[1], 0.0]]
        dist_origin += rad
    
    for level in range(1, levels):
        
        if level > 1 and pts_increase_mult is not None:
            pts_level *= pts_increase_mult

        if level > 1 and pts_increase_add is not None:
            pts_level += pts_increase_add
            
        rad = rad_increase**level * smallest_rad
        dist_origin += rad + dist
        phis = np.linspace(0, 2*np.pi, pts_level, endpoint=False)
        vals = np.linspace(val[0], val[1], pts_level)
                
        ellipses += [[v, rad, rad, 
                      o[0]+np.cos(phi)*dist_origin, 
                      o[1]+np.sin(phi)*dist_origin, 0.0]
                     for phi, v in zip(phis, vals)]
        dist_origin += rad
    
    return odl.phantom.ellipsoid_phantom(space, ellipses)


def resolution_phantom(case=1, nsamples=1000, highres=None):    
    
    if highres is None:
        highres = nsamples
    
    space = odl.uniform_discr([-1, -1], [1, 1], [highres, highres])
    r = .7       
    min_pt = [-r, -r]
    max_pt = [r, r]    
    rad = 0.03
    center = [0, 0]
    pts_level = 5
    pts_increase_add = 2
    plot_center = False

    x, y = space.meshgrid
        
    if case == 1:
        bg = space.element(x + 2 * y)
        bg_val = [.1, 1]
        levels = 5
        val = [-.4, .4]
        
    else:
        bg = space.element(x * (y ** 2))
        bg_val = [0, 1]
        levels = 4
        val = 1
        
    def make_binary(u, threshold=0.5):
        u[u.ufuncs.less_equal(threshold)] = 0
        u[u.ufuncs.greater(threshold)] = 1        

    bg1 = odl.phantom.cuboid(space, min_pt, max_pt)
    bg1[:] = rotate(bg1, 22.5, reshape=False)
    make_binary(bg1)

    bg2 = bg1.space.element(rotate(bg1, 45, reshape=False))
    make_binary(bg2)
    
    u_shape = bg1 * bg2

    def normalize_linar(u, val=[0, 1]):
        u -= u.ufuncs.min()
        u /= u.ufuncs.max()
        u[:] = (val[1] - val[0]) * u + val[0]
    
    if not isinstance(bg_val, list):
        bg_val = [bg_val, bg_val]
            
    u = u_shape * bg
    normalize_linar(u, bg_val)
    u *= u_shape

    res_pattern = get_resolution_pattern(
            space, rad, center, levels=levels, pts_level=pts_level,
            pts_increase_add=pts_increase_add, val=val, 
            plot_center=plot_center)
    
    if case == 1:
        u += res_pattern
    else:
        u *= (1-res_pattern)
            
    
    space2 = odl.uniform_discr([-1, -1], [1, 1], [nsamples, nsamples])
    u = space2.element(resize(u, [nsamples, nsamples]))
    
    u.ufuncs.maximum(0, out=u)
    u.ufuncs.minimum(1, out=u)
    
    return u


class Subsampling(odl.Operator):
    
    def __init__(self, domain, factor=4):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import application
        >>> X = odl.uniform_discr([0, 0], [1, 1], [8, 8])
        >>> S = application.Subsampling(X, 4)
        >>> x = X.one()
        >>> y = S(x)
        """
        
        self.factor = int(factor)
        shape = (np.array(domain.shape) / self.factor).astype('int')
        
        range = odl.uniform_discr(domain.min_pt, domain.max_pt, shape)
                
        super(Subsampling, self).__init__(domain=domain, range=range, 
                                          linear=True)
        
    def _call(self, x, out):
        out[:] = block_reduce(x, block_size=(self.factor, self.factor), 
                              func=np.mean)
                            
    @property
    def adjoint(self):
        op = self
            
        class SubsamplingAdjoint(odl.Operator):
            
            def __init__(self, domain, range):
                """TBC
        
                Parameters
                ----------
                TBC
        
                Examples
                --------        
                >>> import odl
                >>> import application
                >>> X = odl.uniform_discr([0, 0], [1, 1], [8, 8])
                >>> S = application.Subsampling(X, 4)
                >>> S.adjoint.adjoint

                >>> import odl
                >>> import application
                >>> X = odl.uniform_discr([0, 0], [1, 1], [8, 8])
                >>> S = application.Subsampling(X, 4)
                >>> x = odl.phantom.white_noise(S.domain)
                >>> y = odl.phantom.white_noise(S.range)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                """
                self.factor = int(range.shape[0] / domain.shape[0])
                
                super(SubsamplingAdjoint, self).__init__(
                        domain=domain, range=range, linear=True)
                    
            def _call(self, x, out):
                out[:] = np.kron(x, np.ones((op.factor, op.factor)))
         
            @property
            def adjoint(self):
                return op
                    
        return SubsamplingAdjoint(self.range, self.domain)


def superresolution():
    sinfo = resolution_phantom(case=1, nsamples=200, highres=800)
    gtruth = resolution_phantom(case=2, nsamples=200, highres=800)
    
    operator = Subsampling(gtruth.space, factor=5)

    data = odl.phantom.white_noise(operator.range, mean=operator(gtruth), 
                                   stddev=0.01, seed=10)

    return gtruth, sinfo, operator, data


def xray(nviews=15, ndectectors=100):
    gtruth = resolution_phantom(case=1, nsamples=200, highres=800)
    sinfo = resolution_phantom(case=2, nsamples=200, highres=800)
 
    angle_partition = odl.uniform_partition(0, np.pi, nviews)
    detector_partition = odl.uniform_partition(-1.5, 1.5, ndectectors)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    operator = odl.tomo.RayTransform(gtruth.space, geometry, impl='astra_cpu')
    
    data = odl.phantom.salt_pepper_noise(operator(gtruth), fraction=0.05, 
                                         seed=10)

    return gtruth, sinfo, operator, data