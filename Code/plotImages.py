import matplotlib.pyplot as plt
import numpy as np


# Function that plots 15 images, each one containing two separate images that are stacked together
# param img1: First list of images
# param img2: First list of images
# param labels: Labels of the images
# param desc: Title of the figure
def plotStackedImages(img1, img2, labels, title):
    # Plotting the denoised images alongside their noise counterparts
    fig = plt.figure(title, figsize=(90, 10))
    d = img1[0].shape[0]
    for i in range(0, 15):
        ax = fig.add_subplot(5, 3, i + 1)
        left_image = img1[i].reshape(d, d)
        right_image = img2[i].reshape(d, d)
        label = labels[i]
        ax.imshow(np.hstack((left_image, right_image)), cmap='bone')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.show()
