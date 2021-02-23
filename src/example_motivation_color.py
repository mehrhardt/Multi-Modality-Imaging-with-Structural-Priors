#%% Create RGB channels for figure 3
from matplotlib.image import imread, imsave 
from skimage.transform import resize
import os

folder_in = '../data'
folder_out = '../pics'

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    
x = imread('{}/bath_screen.png'.format(folder_in))[:,:,:3]

x /= x.max()
x = resize(x, (500, 900))

imsave('{}/bath_rgb.png'.format(folder_out), x)

red = x.copy()
red[:,:,1:] = 0
imsave('{}/bath_red.png'.format(folder_out), red)

green = x.copy()
green[:,:,0] = 0
green[:,:,2] = 0
imsave('{}/bath_green.png'.format(folder_out), green)

blue = x.copy()
blue[:,:,:2] = 0
imsave('{}/bath_blue.png'.format(folder_out), blue)