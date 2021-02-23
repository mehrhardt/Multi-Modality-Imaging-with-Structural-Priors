#%% Create hyperspectral images for figure 4
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import os

folder_in = '../data'
folder_out = '../pics'

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

input_fname = 'rs_sample02'
sinfo_fname = 'rs_sample02.tif'

data_obj = envi.open('{}/{}.hdr'.format(folder_in, input_fname),
                     '{}/{}.bil'.format(folder_in, input_fname))
data_raw = np.array(data_obj.load())
sinfo_raw = plt.imread('{}/{}'.format(folder_in, sinfo_fname)).astype('float')

for ichannel in [1, 60, 125]:
    fov = [50, 100, 25, 75]
    output_fname = '{}_ch{}_{}-{}x{}-{}'.format(input_fname, ichannel,
                                                   fov[0], fov[1], fov[2], fov[3])
    
    sdata = data_raw.shape[:2]
    s = 4
    sinfo_color = sinfo_raw[fov[0] * s:fov[1] * s, fov[2] * s:fov[3] * s, :]
    sinfo = np.mean(sinfo_color, 2)
    
    data = data_raw[fov[0]:fov[1], fov[2]:fov[3], ichannel]
    
    data /= data.max()
    sinfo_color /= sinfo_color.max()
    sinfo /= sinfo.max()
    
    prefix = '{}/{}'.format(folder_out, output_fname)
    
    if ichannel == 1:
        plt.imsave(prefix + '_sinfo_color.png', sinfo_color)
    plt.imsave(prefix + '_data.png', data, cmap='viridis')