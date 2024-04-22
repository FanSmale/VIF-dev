import matplotlib.pyplot as plt

def v_show(*args):
    num_v_s = len(args)
    num_rows = 1 if num_v_s == 1 else 2
    num_cols = num_v_s // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5.8, 6), dpi=100)
    
    if num_rows == 1:
        axs.imshow(args[0], extent=[0, 0.7, 0.7, 0])
    elif num_cols == 1:
        for i, v in enumerate(args):
            axs[i // num_cols].imshow(v, extent=[0, 0.7, 0.7, 0])
    else:    
        for i, v in enumerate(args):
            axs[i // num_cols, i % num_cols].imshow(v, extent=[0, 0.7, 0.7, 0])
    
    # if num_rows > 1:
    #     fig.tight_layout()
        
    plt.show()

##################################
import cv2
import numpy as np
    
def contours(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    # canny = cv2.Canny(norm_image_to_255, 100, 150)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny

##################################
def fft(images):
    for i, image in enumerate(images):
        dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        images[i] = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1e-8)
    return images

##################################
import re

def extract_number(filename):
    # Use regular expressions to extract the numeric part from the file name.
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return 0
    
##################################
def plot_line(*ys):
    for i, y in enumerate(ys):
        x = np.arange(len(y))
        plt.plot(x, y, label='curve {}'.format(i+1))
        
    plt.title('loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    # Test
    data = np.load('results/t2_id2.npz')
    v_show(*data.values())
    
    # Test
    data = np.load('results/loss_t2_epochs600_1705604004.npy')
    plot_line(data)