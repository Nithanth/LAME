import argparse
import torch
import glob
import pandas as pd
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from model import build_model
from sklearn.metrics import f1_score

def main(image_type, type_img):
    if image_type not in type_img:
        print("Unknown Image_type")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sign_names_df = pd.read_csv('../input/signnames.csv')
    class_names = sign_names_df.SignName.tolist()
    gt_df = pd.read_csv('../input/GTSRB_Final_Test_GT/GT-final_test.csv', delimiter=',')
    gt_df = gt_df.set_index('Filename', drop=True)
    model = build_model(pretrained=False, fine_tune=False, num_classes=43).to(device)
    model.eval()
    model.load_state_dict(torch.load('../outputs/model.pth', map_location=device)['model_state_dict'])
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    all_images = glob.glob(f'../input/GTSRB_Final_Test_Images/GTSRB/Final_Test/{image_type}/*.ppm')
    y_true = []
    y_pred = []

    for image_path in all_images:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1)
        image_name = os.path.basename(image_path)
        gt_idx = gt_df.loc[image_name].ClassId
        y_true.append(gt_idx)
        y_pred.append(class_idx.item())

    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the type of image")
    parser.add_argument("image_type", help="Specify the image type (e.g., baseline_original_images)")
    args = parser.parse_args()
    list_operations = ["test","baseline_original_images","baseline_gaussian_blur","baseline_fgsm_attack","baseline_pgd_attack", "de-gaussian_blur","de-fgsm_attack","de-pgd_attack","de-fgsm_attack_light","de-pgd_attack_light","de-gaussian_blur_light","baseline_fgsm_attack_color_avg", "de-fgsm_attack_color_avg", "de-pgd_attack_color_avg", "baseline_gaussian_blur_color_avg","de-fgsm_attack_light_color_avg","de-pgd_attack_light_color_avg", "baseline_original_images_color_avg","de-gaussian_blur_color_avg", "baseline_pgd_attack_color_avg", "de-gaussian_blur_light_color_avg"]
    main(args.image_type, list_operations)
