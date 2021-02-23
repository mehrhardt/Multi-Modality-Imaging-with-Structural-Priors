#%% Some functionality used in other files to store, save and visualize data
from matplotlib import pyplot as plt
import numpy as np
from odl.contrib.fom import psnr, ssim
from odl.solvers import Callback

# define a function to compute statistic during the iterations
class MyCallback(Callback):

    def __init__(self, iter_save=[], iter_plot=[], prefix=None, obj_fun=None, 
                 gtruth=None, error=False):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter_count = 0
        self.error = error
        
        if prefix is None:
            prefix = ''
        else:
            prefix += '_'
            
        self.prefix = prefix
        self.obj = []
        self.gtruth = gtruth
        self.obj_fun = obj_fun

    def __call__(self, x, **kwargs):

        if len(x) == 2:
            x = x[0]

        k = self.iter_count

        if k in self.iter_save:
            if self.obj_fun is not None:
                self.obj.append(self.obj_fun(x))

        if k in self.iter_plot:
            name = '{}{:04d}'.format(self.prefix, k)
            
            gtruth = self.gtruth
            if gtruth is not None:
                name += '_psnr{:.1f}db_ssim{:0.4f}'.format(psnr(x, gtruth),
                                                           ssim(x, gtruth))
                
            save_result(x, name)            
            if self.error and gtruth is not None:
                save_error(x - gtruth, name + '_error')

        self.iter_count += 1
                        
def save_result(x, filename):
    plt.imsave(filename + '.png', x, cmap='gray', vmin=0, vmax=1)
    
def save_image(x, filename):
    plt.imsave(filename + '.png', x, cmap='gray')

def save_error(x, filename):
    plt.imsave(filename + '.png', x, cmap='RdBu', vmin=-.5, vmax=.5)
            
def save_vfield(x, filename, period_phase=np.pi):
    v = vfield2rgb(x, period_phase=period_phase)
    plt.imsave(filename + '.png', v)

def save_vfield_cmap(filename, period_phase=np.pi):
    t = np.linspace(-1, 1, 100)
    x = np.meshgrid(t, t)
    v = vfield2rgb(np.array(x), period_phase=period_phase)
    plt.imsave(filename + '.png', v)

def vfield2rgb(x, brightness_max=1, period_phase=2*np.pi, phase_shift=0):
    # Computes a colored image that shows a complex image where the magnitude
    # determines the brightness and the phase determines the colours.
    #
    # Arguments:    
    #    x [array] : vector field of size 2x M x N
    #
    #    brightness_max [float; optional] : maximal brightness
    #
    #    period_phase [float; optional] : period of the phase
    #
    #    phase_shift [float; optional] : which colour corresponds to zero phase
    #
    # Output:
    #    out [array] : colour image
    
    x = x[0,:,:] + 1j * x[1,:,:]
    
    magn_x = np.abs(x)
    phas_x = np.angle(x) + np.pi + phase_shift
    phas_x = np.mod(phas_x, period_phase) / period_phase
        
    colour_map=plt.get_cmap('hsv')
                        
    brightness = np.minimum(magn_x, brightness_max) / brightness_max
          
    colour_xx = np.zeros((x.shape[0], x.shape[1], 3))
    
    for i in range(3):
        colour_xx[:,:,i] = colour_map(phas_x)[:,:,i] * brightness
        
    return colour_xx