#%% Run code to reproduce all figures
import os
import src.application as application
import src.misc as misc
import src.models as models

folder_out = 'pics'

if not os.path.exists(folder_out):
    os.mkdir(folder_out)

# images for figure 8
gtruth, sinfo, operator, data = application.superresolution()
prefix = folder_out + '/superresolution_'
misc.save_result(gtruth, prefix + 'groundtruth')
misc.save_result(sinfo, prefix + 'sideinfo')
misc.save_image(data, prefix + 'data')

gtruth, sinfo, operator, data = application.xray()
prefix = folder_out + '/xray_'
misc.save_result(gtruth, prefix + 'groundtruth')
misc.save_result(sinfo, prefix + 'sideinfo')
misc.save_image(data, prefix + 'data')

# images for figures 6 and 7
for i in range(2):
    if i == 0:
        image = gtruth
        simage = 'gtruth'
    elif i == 1:
        image = sinfo
        simage = 'sinfo'
    else: 
        image = None
        simage = None
        
    for mode in ['location', 'direction']:
        for eta in [1e-2, 1e-1, 1e-0]:
            prefix_ = '{}{}_{}_e{:.1e}'.format(prefix, mode, simage, eta)
            
            grad = models.gradient(gtruth.space, sinfo=image, mode=mode, eta=eta, 
                                   show_sinfo=True, prefix=prefix_)

%run example_xray.py
%run example_superresolution.py