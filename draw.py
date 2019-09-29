from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io


if __name__ == '__main__':
    # all train
    sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']
    seq = 'ETH-Bahnhof'
    
    colours = np.random.rand(32,3)
    phase = 'train'

    if not os.path.exists('mot_benchmark'):
        print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n')
        exit()
    plt.ion()
    fig = plt.figure() 

    seq_dets = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    with open('output/%s.txt'%(seq),'w') as out_file:
        print("Processing %s."%(seq))
        for frame in range(int(seq_dets[:,0].max())):
            frame += 1 #detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:,0]==frame,1:6]
            
            ax1 = fig.add_subplot(111, aspect='equal')
            fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
            im =io.imread(fn)
            ax1.imshow(im)
            plt.title(seq+' Tracked Targets')

            for d in dets:
                d = d.astype(np.int32)
                id_, x1, y1, w, h = d
                ax1.add_patch(patches.Rectangle((x1,y1),w,h,fill=False,lw=3,ec=colours[id_%32,:]))
                ax1.set_adjustable('box-forced')

            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()