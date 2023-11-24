from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the image and convert it to grayscale
def load_image(image_path):
    return np.array(Image.open(image_path).convert('1'))

def convolve_filter(image, filter, stride=1, padding=0):
    # Add zero padding to the image such that the output image will have the same size
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Get dimensions of the image and the filter
    img_height, img_width = image.shape
    filter_height, filter_width = filter.shape

    # Determine the size of the output (convolved) image taking into account the stride
    output_height = (img_height - filter_height) // stride + 1
    output_width = (img_width - filter_width) // stride + 1

    # Create an empty output (convolved) image
    output = np.zeros((output_height, output_width))

    # Iterate over the image to apply the convolution
    for i in range(0, output_height):
        for j in range(0, output_width):
            # Determine the starting indices for the region in the original image
            start_i = i * stride
            start_j = j * stride

            # Extract the region of interest from the image that corresponds to the filter's size
            region = image[start_i:start_i+filter_height, start_j:start_j+filter_width]
            
            # Perform element-wise multiplication and sum it up to get the convolved value
            conv_value = np.sum(region * filter)
            
            # Assign the convolved value to the output
            output[i, j] = conv_value

    return output


if __name__ == "__main__":
    image = load_image("example.jpg")
    filter = np.ones((7,7))
    filter[:,3:] = 0
    for _ in tqdm(range(5)):
        image = convolve_filter(image, filter, stride=1, padding=6)
        image = image -np.min(image)
        image = image / np.max(image)
        plt.imshow(image, cmap='gray')
        plt.show()
    #output = convolve_filter(output, filter)
    plt.imshow(image, cmap='gray')
    plt.show()