from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def scale_to_255(tif_img):
    np_img = np.asarray(tif_img, dtype=np.float32)
    np_img = np_img - np_img.min()
    np_img = np_img / np_img.max()
    # scale to int and 255
    np_img = np_img * 255
    np_img = np_img.astype(np.int8)
    return Image.fromarray(np_img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth") # Change to sam_vit_b... for the smaller model
predictor = SamPredictor(sam)

def get_mask(image_path, pos_positions, neg_positions):
    if image_path.endswith(".tif") or image_path.endswith(".tiff"):
        image = Image.open(image_path)
        scaled_image_arr = scale_to_255(image)
        scaled_image_arr = scaled_image_arr.convert("RGB")
        image = np.asarray(scaled_image_arr)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    predictor.set_image(image)

    # Straight line down the middle
    # Right, down

    height, width, _ = image.shape


    center_x = width // 2

    # Define the y-coordinates for the dots
    upper_third = height // 5
    middle = height // 2
    lower_third = 4 * height // 5
    offset = int(.10 * width)

    # Define the dot positions
    input_point = np.array(pos_positions + neg_positions)
    #input_point = np.array([(center_x, middle)])
    #input_point = np.array([(center_x, middle)])
    input_label = np.array([1] * len(pos_positions) + [0] * len(neg_positions))

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.title(f"Area = {masks.sum()}")
    plt.show() 

#get_mask("example.jpg")