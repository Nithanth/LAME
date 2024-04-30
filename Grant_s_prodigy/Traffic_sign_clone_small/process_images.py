import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import label
import os
import glob

def process_image(original_img, labeled_img):
    output_img = np.copy(original_img)
    unique_colors = np.unique(labeled_img.reshape(-1, labeled_img.shape[2]), axis=0)
    for color in unique_colors:
        mask = np.all(labeled_img == color, axis=2)
        structure = np.ones((3, 3))
        labeled, num_features = label(mask, structure)
        for feature_id in range(1, num_features + 1):
            feature_mask = labeled == feature_id
            if np.any(feature_mask):
                original_pixels = original_img[feature_mask]
                mean_color = np.mean(original_pixels, axis=0).astype(np.uint8)
                output_img[feature_mask] = mean_color
    return output_img

def process_folder(folder_path):
    base_path = "/Users/gman/Desktop/Traffic_sign_clone_small/input/GTSRB_Final_Test_Images/GTSRB/Final_Test"
    image_dir = os.path.join(base_path, folder_path)
    save_dir = os.path.join(image_dir + "_color_avg")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_path in glob.glob(os.path.join(image_dir, '*.ppm')):
        img_name = os.path.basename(image_path)
        im = cv2.imread(image_path)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        reshaped_im = im.reshape((-1, 3))
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(reshaped_im)
        dominant_colors = kmeans.cluster_centers_.astype('uint8')
        labels = kmeans.labels_.reshape(im.shape[0], im.shape[1])
        new_img = np.zeros_like(im)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                new_img[i, j] = dominant_colors[labels[i, j]]

        final_img = process_image(im, new_img)
        save_path = os.path.join(save_dir, img_name)
        final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, final_img_bgr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process_images.py <folder_name>")
        sys.exit(1)
    folder_name = sys.argv[1]
    process_folder(folder_name)
