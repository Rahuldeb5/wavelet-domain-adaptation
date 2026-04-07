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
from torchvision.transforms import ToTensor
import tifffile as tif
import pandas as pd
from wavelet import compute_mean_LL, wavelet_adapt
from fourier import compute_mean_amplitude, fourier_adapt
from utils import cosine_similarity, l2_distance, mean_pixel_diff, histogram_intersection
from edge_analysis import analyze_pair
from config import RANDOM_SEED, IMG_PATH, IMG_LIMIT, TEST_SIZE, BATCH_SIZE, LR, WEIGHT_DECAY, DEVICE

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

to_tensor = ToTensor()

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
    csv_dir = f"../results/baseline"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/baseline.csv"
    
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("source"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    done.add((parts[0].strip(), parts[1].strip()))
    else:
        with open(csv_path, "w") as f:
            f.write("source_region,target_region,precision,recall,f1,iou,miou,oa\n")

    for source_region in regions_dict:
        all_done = all(
            (source_region, t) in done
            for t in regions_dict if t != source_region
        )
        if all_done:
            continue
        
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

            with open(csv_path, "a") as f:
                f.write(f"{source_region},{target_region},{precision:.4f},{recall:.4f},{f1:.4f},{iou:.4f},{miou:.4f},{oa:.4f}\n")

            print(f"{source_region} -> {target_region}: IoU={iou:.4f}")
            
def wavelet(alpha):
    csv_dir = "../results/wavelet"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/wavelet_a{int(alpha*100)}.csv"

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("source"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    done.add((parts[0].strip(), parts[1].strip()))
    else:
        with open(csv_path, "w") as f:
            f.write("source_region,target_region,precision,recall,f1,iou,miou,oa\n")

    for target_region in regions_dict:
        tgt_images = regions_dict[target_region]
        tgt_tensors = [to_tensor(tif.imread(p)).to(DEVICE) for p in tgt_images]
        mean_ll = compute_mean_LL(tgt_tensors)

        tgt_test_mask = [f.replace("img", "mask") for f in tgt_images]
        test_dataset = LandslideDataset(tgt_images, tgt_test_mask)
        testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        for source_region in regions_dict:
            if source_region == target_region:
                continue
            if (source_region, target_region) in done:
                continue

            source_images = regions_dict[source_region]
            source_masks = [f.replace("img", "mask") for f in source_images]

            train_img, val_img, train_mask, val_mask = train_test_split(
                source_images, source_masks, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )

            adapted_imgs = []
            for p in train_img:
                src_tensor = to_tensor(tif.imread(p)).to(DEVICE)
                adapted = wavelet_adapt(src_tensor, mean_ll, alpha)
                adapted_imgs.append(adapted.cpu())

            train_dataset = AdaptedDataset(adapted_imgs, train_mask)
            val_dataset = LandslideDataset(val_img, val_mask)
            trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            valLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = ResNetUNet().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            train_model(model, optimizer, trainLoader, valLoader, source_region, target_region)

            precision, recall, f1, iou, miou, oa = test_model(model, testLoader)

            with open(csv_path, "a") as f:
                f.write(f"{source_region},{target_region},{precision:.4f},{recall:.4f},{f1:.4f},{iou:.4f},{miou:.4f},{oa:.4f}\n")

            print(f"{source_region} -> {target_region}: IoU={iou:.4f}")

def fourier(beta):
    csv_dir = "../results/fourier"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/fourier_b{int(beta*100)}.csv"

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("source"):
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    done.add((parts[0].strip(), parts[1].strip()))
    else:
        with open(csv_path, "w") as f:
            f.write("source_region,target_region,precision,recall,f1,iou,miou,oa\n")

    for target_region in regions_dict:
        tgt_images = regions_dict[target_region]
        tgt_tensors = [to_tensor(tif.imread(p)).to(DEVICE) for p in tgt_images]
        mean_amp = compute_mean_amplitude(tgt_tensors)

        tgt_test_mask = [f.replace("img", "mask") for f in tgt_images]
        test_dataset = LandslideDataset(tgt_images, tgt_test_mask)
        testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        for source_region in regions_dict:
            if source_region == target_region:
                continue
            if (source_region, target_region) in done:
                continue

            source_images = regions_dict[source_region]
            source_masks = [f.replace("img", "mask") for f in source_images]

            train_img, val_img, train_mask, val_mask = train_test_split(
                source_images, source_masks, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )

            adapted_imgs = []
            for p in train_img:
                src_tensor = to_tensor(tif.imread(p)).to(DEVICE)
                adapted = fourier_adapt(src_tensor, mean_amp, beta)
                adapted_imgs.append(adapted.cpu())

            train_dataset = AdaptedDataset(adapted_imgs, train_mask)
            val_dataset = LandslideDataset(val_img, val_mask)
            trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            valLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = ResNetUNet().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            train_model(model, optimizer, trainLoader, valLoader, source_region, target_region)

            precision, recall, f1, iou, miou, oa = test_model(model, testLoader)

            with open(csv_path, "a") as f:
                f.write(f"{source_region},{target_region},{precision:.4f},{recall:.4f},{f1:.4f},{iou:.4f},{miou:.4f},{oa:.4f}\n")

            print(f"{source_region} -> {target_region}: IoU={iou:.4f}")

def similarity():
    csv_dir = "../results/similarity"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/pairwise_similarity.csv"

    with open(csv_path, "w") as f:
        f.write("source_region,target_region,cosine_sim,l2_dist,mean_pixel_diff,hist_intersection\n")

    print("Computing mean LL per region...")
    mean_lls = {}
    for region, paths in regions_dict.items():
        tgt_tensors = [to_tensor(tif.imread(p)).to(DEVICE) for p in paths]
        mean_ll = compute_mean_LL(tgt_tensors).squeeze(0).cpu()
        mean_lls[region] = mean_ll
        print(f"  {region}: {mean_ll.shape}")

    print("Computing pairwise metrics...")
    for src in regions_dict:
        for tgt in regions_dict:
            if src == tgt:
                continue

            cos = cosine_similarity(mean_lls[src], mean_lls[tgt])
            l2 = l2_distance(mean_lls[src], mean_lls[tgt])
            mpd = mean_pixel_diff(mean_lls[src], mean_lls[tgt])
            hi = histogram_intersection(mean_lls[src], mean_lls[tgt])

            with open(csv_path, "a") as f:
                f.write(f"{src},{tgt},{cos:.6f},{l2:.6f},{mpd:.6f},{hi:.6f}\n")

            print(f"{src} -> {tgt}: cos={cos:.4f}, l2={l2:.2f}, mpd={mpd:.4f}, hi={hi:.4f}")

def edge_distortion(alpha=0.40, beta=0.15):    
    csv_dir = "../results/edge_analysis"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/edge_metrics_b{(int)(beta*100)}.csv"

    with open(csv_path, "w") as f:
        f.write("source_region,target_region,image_idx,"
                "wav_hf_ratio,fda_hf_ratio,"
                "wav_edge_sim,fda_edge_sim,"
                "wav_LH_corr,wav_HL_corr,wav_HH_corr,"
                "fda_LH_corr,fda_HL_corr,fda_HH_corr\n")

    for target_region in regions_dict:
        tgt_images = regions_dict[target_region]
        tgt_tensors = [to_tensor(tif.imread(p)).to(DEVICE) for p in tgt_images]
        mean_ll = compute_mean_LL(tgt_tensors)
        mean_amp = compute_mean_amplitude(tgt_tensors)

        for source_region in regions_dict:
            if source_region == target_region:
                continue

            src_paths = regions_dict[source_region][:5]
            for i, p in enumerate(src_paths):
                src_tensor = to_tensor(tif.imread(p)).to(DEVICE)
                
                results = analyze_pair(src_tensor, mean_ll, mean_amp, alpha, beta)

                with open(csv_path, "a") as f:
                    f.write(f"{source_region},{target_region},{i},"
                            f"{results['wav_hf_ratio']:.6f},{results['fda_hf_ratio']:.6f},"
                            f"{results['wav_edge_sim']:.6f},{results['fda_edge_sim']:.6f},"
                            f"{results['wav_LH_corr']:.6f},{results['wav_HL_corr']:.6f},{results['wav_HH_corr']:.6f},"
                            f"{results['fda_LH_corr']:.6f},{results['fda_HL_corr']:.6f},{results['fda_HH_corr']:.6f}\n")

            print(f"{source_region} -> {target_region}: done")
    
    df = pd.read_csv(csv_path)
    print("\n" + "="*60)
    print("EDGE DISTORTION SUMMARY")
    print("="*60)
    print(f"  Wavelet edge sim:  {df['wav_edge_sim'].mean():.4f} ± {df['wav_edge_sim'].std():.4f}")
    print(f"  FDA edge sim:      {df['fda_edge_sim'].mean():.4f} ± {df['fda_edge_sim'].std():.4f}")
    print(f"  Wavelet HF ratio:  {df['wav_hf_ratio'].mean():.4f} (should be ~1.0)")
    print(f"  FDA HF ratio:      {df['fda_hf_ratio'].mean():.4f}")
    print(f"  FDA LH corr:       {df['fda_LH_corr'].mean():.4f}")
    print(f"  FDA HL corr:       {df['fda_HL_corr'].mean():.4f}")
    print(f"  FDA HH corr:       {df['fda_HH_corr'].mean():.4f}")

if __name__ == "__main__":
    import argparse
    import sys
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.10)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("../logs", exist_ok=True)
    sys.stdout = open(f"../logs/{args.method}_{timestamp}.log", "w", buffering=1)

    if args.method == "baseline":
        baseline()
    elif args.method == "wavelet":
        wavelet(alpha=args.alpha)
    elif args.method == "fourier":
        fourier(beta=args.beta)
    elif args.method == "similarity":
        similarity()
    elif args.method == "edge":
        edge_distortion()

