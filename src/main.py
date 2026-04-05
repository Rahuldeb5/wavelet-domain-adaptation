import os
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ResNetUNet
from dataset import LandslideDataset, AdaptedDataset
from train import train_model, test_model
import numpy as np
from config import RANDOM_SEED, IMG_PATH, IMG_LIMIT, TEST_SIZE, BATCH_SIZE, LR, WEIGHT_DECAY, DEVICE

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

regions = [f for f in os.listdir(IMG_PATH) if (os.path.isdir(os.path.join(IMG_PATH, f)) and f != "study areas shp")]

regions_dict = dict()

for region in regions:
    dataset_dir = IMG_PATH + region
    image_dir = os.path.join(dataset_dir, "img")
    img_list = os.listdir(image_dir)

    all_images = sorted(os.path.join(image_dir, f) for f in img_list)

    if len(all_images) > IMG_LIMIT:
        all_images = random.sample(all_images, IMG_LIMIT)

    regions_dict[region] = all_images

def baseline():
    output = "source_region,target_region,precision,recall,f1,iou,miou,oa"
    
    csv_dir = f"../../results/baseline"
    os.makedirs(csv_dir, exist_ok=True)

    for source_region in regions_dict:
        source_images = regions_dict[source_region]
        source_masks = [f.replace("img", "mask") for f in source_images]

        train_img, val_img, train_mask, val_mask = train_test_split(
            source_images, source_masks, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        
        model = ResNetUNet().to(DEVICE)
        
        train_dataset = LandslideDataset(train_img, train_mask)
        val_dataset = LandslideDataset(val_img, val_mask)
        trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        valLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train_model(model, optimizer, trainLoader, valLoader, source_region)

        for target_region in regions_dict:
            if source_region == target_region:
                continue
            
            tgt_test_img = regions_dict[target_region]
            tgt_test_mask = [f.replace("img", "mask") for f in tgt_test_img]

            test_dataset = LandslideDataset(tgt_test_img, tgt_test_mask)
            testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            precision, recall, f1, iou, miou, oa = test_model(model, testLoader)

            result = f"{source_region}, {target_region}, {precision:.4f}, {recall:.4f}, {f1:.4f}, {iou:.4f}, {miou:.4f}, {oa:.4f}"
            output += f"\n{result}"

    csv_path = f"{csv_dir}/baseline.csv"
    with open(csv_path, "w") as f:
        f.write(output)

if __name__ == "__main__":
    import argparse
    import sys
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    sys.stdout = open(f"../logs/{args.method}_{timestamp}.log", "w", buffering=1)

    if args.method == "baseline":
        baseline()
