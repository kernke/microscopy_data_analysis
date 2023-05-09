# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:32:54 2023

@author: kernke
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

#%% make_scale_bar
def vis_make_scale_bar(
    images, pixratios, lengthperpix, barlength, org, thickness=4, color=(255, 0, 0)
):

    org = np.array(org)
    for i in range(len(images)):
        pixlength = (barlength / lengthperpix) / pixratios[i]
        pixlength = np.round(pixlength).astype(int)
        pt2 = org + np.array([0, pixlength])
        cv2.line(images[i], org[::-1], pt2[::-1], color, thickness=thickness)



#%% make_mp4
def vis_make_mp4(filename, images, fps):

    with imageio.get_writer(filename, mode="I", fps=fps) as writer:
        for i in range(len(images)):
            writer.append_data(images[i])

    return True


#%% zoom


def vis_zoom(img, zoom_center, final_height, steps, gif_resolution_to_final=1):

    iratio = img.shape[0] / img.shape[1]

    final_size = np.array([final_height, final_height / iratio])

    startpoints = np.zeros([4, 2], dtype=int)
    endpoints = np.zeros([4, 2], dtype=int)

    startpoints[2, 1] = img.shape[1] - 1
    startpoints[1, 0] = img.shape[0] - 1
    startpoints[3] = img.shape
    startpoints[3] -= 1

    endpoints[0] = np.round(zoom_center - final_size / 2).astype(int)
    endpoints[3] = np.round(zoom_center + final_size / 2).astype(int)

    tocorner = np.array([-final_size[0], final_size[1]]) / 2
    endpoints[1] = np.round(zoom_center - tocorner).astype(int)
    tocorner = np.array([final_size[0], -final_size[1]]) / 2
    endpoints[2] = np.round(zoom_center - tocorner).astype(int)

    steps += 1
    cornerpoints = np.zeros([steps, 4, 2], dtype=int)
    pixratios = np.zeros(steps)
    for i in range(4):
        for j in range(2):
            cornerpoints[:, i, j] = np.round(
                np.linspace(startpoints[i, j], endpoints[i, j], steps)
            ).astype(int)

    final_resolution = np.round(final_size * gif_resolution_to_final).astype(int)
    images = np.zeros([steps, final_resolution[0], final_resolution[1]])
    for i in range(steps):
        pixratios[i] = (
            cornerpoints[i, 1, 0] - cornerpoints[i, 0, 0]
        ) / final_resolution[0]

        roi_img = img[
            cornerpoints[i, 0, 0] : cornerpoints[i, 1, 0],
            cornerpoints[i, 0, 1] : cornerpoints[i, 2, 1],
        ]

        ratio = roi_img.shape[0] / final_resolution[0]  # size
        sigma = ratio / 4

        ksize = np.round(5 * sigma).astype(int)
        if ksize % 2 == 0:
            ksize += 1
        roi_img = cv2.GaussianBlur(roi_img, [ksize, ksize], sigma)
        images[i] = cv2.resize(roi_img, final_resolution[::-1], cv2.INTER_AREA)

    return images, pixratios

#%%  plot_sortout
def vis_plot_sortout(image, sortout, legend=True, alpha=0.5, markersize=0.5):
    plt.imshow(image, cmap="gray")
    colors = ["b", "r", "g", "c", "m", "y"]
    for j in range(len(sortout)):
        count = 0
        for i in sortout[j]:
            if count == 0:
                plt.plot(
                    i[:, 1],
                    i[:, 0],
                    "o",
                    c=colors[j],
                    alpha=alpha,
                    label=str(j),
                    markersize=markersize,
                )
            else:
                plt.plot(
                    i[:, 1],
                    i[:, 0],
                    "o",
                    c=colors[j],
                    alpha=alpha,
                    markersize=markersize,
                )
            count += 1
    if legend == True:
        plt.legend()
