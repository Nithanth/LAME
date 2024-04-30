import argparse
import numpy as np
import cv2
import torch
import glob as glob
import pandas as pd
import os
import albumentations as A
import time
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
import torch.nn as nn
from torch import topk
from model import build_model
import sys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


def make_ppm(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(source_dir, filename)
            with Image.open(file_path) as img:    
                output_filename = os.path.splitext(filename)[0] + '.ppm'
                output_path = os.path.join(output_dir, output_filename)
                img.save(output_path, 'PPM')
    print("Conversion complete.")
    

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ChannelReducer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return self.conv(x)
    
def create_canny_images(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    blurred = cv2.GaussianBlur(grayscale, (5,5), 0) # apply Gaussian blur
    canny = cv2.Canny(blurred, 30, 150)
    return canny
    
def main(image_type,type_img):
    print("Capturing images of type:", image_type)
    if image_type in type_img:
        print(f"Image_type was found: {image_type}")
    else:
        print("Unknown Image_type")
        sys.exit()


    # Define computation device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class names.
    sign_names_df = pd.read_csv('../input/signnames.csv')
    class_names = sign_names_df.SignName.tolist()

    # DataFrame for ground truth.
            # '../input/GTSRB_Final_Test_GT/GT-final_test_TEST.csv', 

    gt_df = pd.read_csv(
       '../input/GTSRB_Final_Test_GT/GT-final_test.csv', 

        delimiter=','
    )
    gt_df = gt_df.set_index('Filename', drop=True)

    # Initialize model, switch to eval model, load trained weights.
    model = build_model(
        pretrained=False,
        fine_tune=False, 
        num_classes=43
    ).to(device)
    model = model.eval()
    model.load_state_dict(
        torch.load(
            '../outputs/model.pth', map_location=device
        )['model_state_dict']
    )

    # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # Generate the class activation maps upsample to 256x256.
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def apply_color_map(CAMs, width, height, orig_image):
        for i, cam in enumerate(CAMs):
            heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.5 + orig_image * 0.5
            result = cv2.resize(result, (224, 224))
            return result

    def visualize_and_save_map(
        result, orig_image, gt_idx=None, class_idx=None, save_name=None
    ):
        # Put class label text on the result.
        if class_idx is not None:
            cv2.putText(
                result, 
                f"Pred: {str(class_names[int(class_idx)])}", (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                cv2.LINE_AA
            )
        if gt_idx is not None:
            cv2.putText(
                result, 
                f"GT: {str(class_names[int(gt_idx)])}", (5, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                cv2.LINE_AA
            )
        orig_image = cv2.resize(orig_image, (224, 224))
        img_concat = cv2.hconcat([
            np.array(result, dtype=np.uint8), 
            np.array(orig_image, dtype=np.uint8)
        ])
        cv2.imshow('Result', img_concat)
        cv2.waitKey(1)
        if save_name is not None:
            cv2.imwrite(f"../outputs/test_results/{image_type}/CAM_{save_name}.jpg", img_concat)
    

    # Hook the feature extractor.
    # https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
        
    model._modules.get('features').register_forward_hook(hook_feature)
    # Get the softmax weight.
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

    # Define the transforms, resize => tensor => normalize.
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
        ])
    
    canny_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485],
            std=[0.229]
        )])

    counter = 0
    # Run for all the test images.
    all_images = glob.glob(f'../input/GTSRB_Final_Test_Images/GTSRB/Final_Test/{image_type}/*.ppm')

    correct_count = 0
    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second. 
    reducer = ChannelReducer(in_channels=4, out_channels=3, kernel_size=3)
    reducer.load_state_dict(torch.load('../../InferenceExperiments/round2.pth'))
    reducer.eval()
    
    all_gt_labels = []
    all_pred_labels = []
    
    for i, image_path in enumerate(all_images):
        # Read the image.
        image = cv2.imread(image_path)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = orig_image.shape
        # Apply the image transforms.
        image_tensor = transform(image=image)['image']

        # #ARTI START
        canny_image = create_canny_images(image)
        canny_image = canny_image.reshape((canny_image.shape[0], canny_image.shape[1], -1))
        canny_image = canny_transform(image=canny_image)['image']
        canny_image = torch.from_numpy(canny_image)
        canny_image = canny_image.permute(2, 0, 1)
        concatenated_image = torch.cat([image_tensor, canny_image], dim=0)
        image_tensor = reducer(concatenated_image)
      
        # # ARTI END
        
        
        # Add batch dimension.
        image_tensor = image_tensor.unsqueeze(0)
        # Forward pass through model.
        start_time = time.time()
        outputs = model(image_tensor.to(device))
        end_time = time.time()
        # Get the softmax probabilities.
        probs = F.softmax(outputs).data.squeeze()
        # Get the class indices of top k probabilities.
        class_idx = topk(probs, 1)[1].int()
        # Get the ground truth.
        image_name = image_path.split(os.path.sep)[-1]
        gt_idx = gt_df.loc[image_name].ClassId
        # Check whether correct prediction or not.
        if gt_idx == class_idx:
            correct_count += 1
        all_gt_labels.append(gt_idx.item())
        all_pred_labels.append(class_idx.item())
        # Generate class activation mapping for the top1 prediction.
        CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
        # File name to save the resulting CAM image with.
        save_name = f"{image_path.split('/')[-1].split('.')[0]}"
        # Show and save the results.
        
        # output_array = output_tensor.squeeze(0).detach().cpu().numpy()
        # output_image = np.transpose(output_array, (1, 2, 0))
        # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        
        # result = apply_color_map(CAMs, width, height, orig_image)
        # #result = apply_color_map(CAMs, width, height, output_image)
        # #visualize_and_save_map(result, output_image, gt_idx, class_idx, save_name)
        # visualize_and_save_map(result, orig_image, gt_idx, class_idx, save_name)
        
        counter += 1
        print(counter)
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1

    print(f"{image_type}:Total number of test images: {len(all_images)}")
    print(f"{image_type}:Total correct predictions: {correct_count}")
    print(f"{image_type}:Accuracy: {correct_count/len(all_images)*100:.3f}")
    f1 = f1_score(all_gt_labels, all_pred_labels, average='weighted')
    print(f"{image_type}:F1 Score: {f1}")

    # Close all frames and video windows.
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the type of attack ")
    parser.add_argument("image_type", help="baseline_original_images, baseline_gaussian_blur, baseline_fgsm_attack, baseline_fgsm_canny_attack, baseline_pgd_attack,de-gaussian_blur,de-fgsm_attack,de-pgd_attack,de-fgsm_attack_light,de-pgd_attack_light,de-gaussian_blur_light")
    args = parser.parse_args()
    list_operations = ["test","gaussian_blur","pgd_attack", "baseline_original_images","baseline_gaussian_blur","stopsigns_only","baseline_fgsm_attack","baseline_fgsm_canny_attack", "baseline_pgd_attack", "de-gaussian_blur","de-fgsm_attack","de-pgd_attack","de-fgsm_attack_light","de-pgd_attack_light","de-gaussian_blur_light"]
    main(args.image_type,list_operations )

     




