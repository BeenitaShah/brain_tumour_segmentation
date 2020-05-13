import numpy as np
import os
import glob
from glob import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from bxs633.evaluation.helper_functions import *


class ScrollMRI(object):
    """ Initialistion method
   :param ax1,ax2,ax3,ax4: sub-plots of a figure
   :param gt: ground truth volume
   :param pred_one: first prediction volume to be visualized
   :param pred_two: second prediction volume to be visualized
   :param pred_three: third prediction volume to be visualized """
    def __init__(self, ax1, ax2, ax3, ax4, gt, pred_one, pred_two, pred_three):
        #colour map mapping class values to colours
        self.colors = ['black', 'indigo', 'tab:blue', '#9CC3D5FF']
        self.levels = [0., 1., 2., 4.]
        self.cmap, self.norm = mpl.colors.from_levels_and_colors(levels=self.levels, colors=self.colors, extend='max')

        #mapping class values to colour for legend
        zero_patch = mpatches.Patch(color='black', label='Background (0)')
        one_patch = mpatches.Patch(color='indigo', label='Necrotic and Non-enhancing Tumor Core (1)')
        two_patch = mpatches.Patch(color='tab:blue', label='Peritumoral Edema (2)')
        four_patch = mpatches.Patch(color='#9CC3D5FF', label='GD-Enhancing Tumor (4)')
        plt.legend(handles=[zero_patch,one_patch,two_patch,four_patch], prop={'size': 6}, loc='upper center', bbox_to_anchor=(-1.4, -0.2), shadow=True, ncol=2)

        self.ax1 = ax1
        self.gt = gt
        self.gt = self.cmap(self.norm(gt))
        _, _, self.slices = gt.shape
        #Initial slice index set to zero.
        self.ind = 0
        self.ax1.set_title('Ground Truth')
        self.ax1.set_axis_off()
        self.im1 = ax1.imshow(self.gt[self.ind])

        self.ax2 = ax2
        self.pred_single = pred_one
        self.pred_single = self.cmap(self.norm(self.pred_single))
        self.ax2.set_title('Co-reg (3 Modalities)')
        self.ax2.set_axis_off()
        self.im2 = ax2.imshow(self.pred_single[self.ind])

        self.ax3 = ax3
        self.pred_coreg_4 = pred_two
        self.pred_coreg_4= self.cmap(self.norm(self.pred_coreg_4))
        self.ax3.set_title('Co-reg (4 Modalities)')
        self.ax3.set_axis_off()
        self.im3 = ax3.imshow(self.pred_coreg_4[self.ind])
        
        
        self.ax4 = ax4
        self.pred_ensemble = pred_three
        self.pred_ensemble = self.cmap(self.norm(self.pred_ensemble))
        self.ax4.set_title('Ensemble (3 Modalities, Voting)')
        self.ax4.set_axis_off()
        self.im4 = ax4.imshow(self.pred_ensemble[self.ind])
        self.update()

    """ Update visualiser upon scroll event
        :param event: scroll event """
    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    """ Update method for all subplots """
    def update(self):
        self.im1.set_data(self.gt[self.ind])
        self.ax1.set_title('Ground Truth')
        self.ax1.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()

        self.im2.set_data(self.pred_single[self.ind])
        self.ax2.set_title('Co-reg (3 Modalities)')
        self.im2.axes.figure.canvas.draw()

        self.im3.set_data(self.pred_coreg_4[self.ind])
        self.ax3.set_title('Co-reg (4 Modalities)')
        self.im3.axes.figure.canvas.draw()

        self.im4.set_data(self.pred_ensemble[self.ind])
        self.ax4.set_title('Ensemble (3 Modalities, Voting)')
        self.im4.axes.figure.canvas.draw()

""" Visualise predictions from 3 models
   :param file_name: name of patient file to visualise - should be present in all model prediction and ground truth directories
   :param gt_dir: ground truth data directory
   :param mode_one_dir: first model prediction directory - 3 modality coreg data trained model
   :param mode_two_dir: second model prediction directory - 4 modality coreg data trained model
   :param mode_three_dir: third model prediction directory - ensemble model predictions"""
def main():
    file_name = sys.argv[1]
    gt_dir = sys.argv[2]
    model_one_dir = sys.argv[3]
    model_two_dir = sys.argv[4]
    model_three_dir = sys.argv[5]

    gt = np.load(gt_dir + file_name)
    gt = gt[0,:,:,:]
    gt = center_crop(gt)

    #Ensemble pred
    ensemble_pred = np.load(model_three_dir + file_name)

    #coreg_3
    coreg_3 = np.load(model_one_dir + file_name)
    coreg_3 = get_labels(coreg_3)

    #coreg_4
    coreg_4 = np.load(model_two_dir + file_name)
    coreg_4 = get_labels(coreg_4)

    mpl.rc('font',family='Comic Sans MS')
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)
    tracker = ScrollMRI(ax1, ax2, ax3, ax4, gt, coreg_3, coreg_4, ensemble_pred)
    fig.suptitle('Brain Tumour Segmentation', fontname='Comic Sans MS', fontsize=20, y=0.8)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()