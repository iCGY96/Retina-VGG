import os
import glob
import shutil
import requests
import zipfile
import json
import numpy as np
import cv2
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from salicon.salicon import SALICON  # Use SALICON API for annotations

###################################################
# Dummy Retina class (example implementation; replace with actual)
###################################################
class DummyRetina:
    def run(self, img):
        self.img = img
    def getParvo(self):
        return self.img

###################################################
# Path utilities: normalization, directory creation, and cleanup
###################################################
def norm_path(path: str) -> str:
    """Return an absolute, normalized path."""
    return os.path.normpath(os.path.abspath(path))

def ensure_dir(path: str) -> str:
    """Ensure that a directory exists (create it if necessary)."""
    path = norm_path(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def remove_tree(path: str):
    """Recursively remove a directory if it exists."""
    path = norm_path(path)
    if os.path.exists(path):
        print(f"Removing directory: {path}")
        shutil.rmtree(path)

def remove_file(path: str):
    """Remove a file if it exists."""
    path = norm_path(path)
    if os.path.exists(path) and os.path.isfile(path):
        print(f"Removing file: {path}")
        os.remove(path)

###################################################
# Download and extraction functions
###################################################
def download_file(url: str, dest_path: str):
    """Download file from a URL to the destination path if not already present."""
    dest_path = norm_path(dest_path)
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists, skipping download.")
        return
    print(f"Downloading from {url} to {dest_path} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")

def extract_zip(zip_path: str, extract_to: str):
    """Extract a ZIP file to the given directory if not already extracted."""
    zip_path = norm_path(zip_path)
    extract_to = ensure_dir(extract_to)
    # If the target directory already contains files, assume extraction is done
    if os.listdir(extract_to):
        print(f"{extract_to} already contains files, skipping extraction.")
        return
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def prepare_dataset(url: str, zip_filename: str, extract_dir: str):
    """Download and extract dataset from a URL if not already done."""
    download_file(url, zip_filename)
    extract_zip(zip_filename, extract_dir)

###################################################
# MIT1003 dataset: Download, unify as test set, and cleanup functions
###################################################
def download_mit1003():
    """Download MIT1003 stimuli images and fixation maps if not already available."""
    # Download stimuli images
    stimuli_url = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip"
    stimuli_zip = norm_path("ALLSTIMULI.zip")
    stimuli_dir = ensure_dir("MIT1003_Stimuli")
    prepare_dataset(stimuli_url, stimuli_zip, stimuli_dir)
    
    # Download fixation maps
    fixmap_url = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"
    fixmap_zip = norm_path("ALLFIXATIONMAPS.zip")
    fixmap_dir = ensure_dir("MIT1003_FixMaps")
    prepare_dataset(fixmap_url, fixmap_zip, fixmap_dir)

def unify_mit1003_as_test():
    """
    Unify MIT1003 data as a test set.
    Organize stimuli images and corresponding fixation maps into data/MIT1003/test.
    """
    stimuli_root = norm_path("MIT1003_Stimuli/ALLSTIMULI")
    fixmaps_root = norm_path("MIT1003_FixMaps/ALLFIXATIONMAPS")
    
    base_data = ensure_dir("data")
    mit1003_dir = ensure_dir(os.path.join(base_data, "MIT1003"))
    test_img_dir  = ensure_dir(os.path.join(mit1003_dir, "test", "images"))
    test_fix_dir  = ensure_dir(os.path.join(mit1003_dir, "test", "fixmaps"))
    
    all_stimuli_paths = sorted([f for f in glob.glob(os.path.join(stimuli_root, "*.jpeg")) if os.path.isfile(f)])
    pairs = []
    for img_path in all_stimuli_paths:
        img_name = os.path.basename(img_path)
        base, ext = os.path.splitext(img_name)
        fixmap_name = base + "_fixMap.jpg"
        fixmap_path = os.path.join(fixmaps_root, fixmap_name)
        if os.path.exists(fixmap_path):
            pairs.append((img_path, fixmap_path))
        else:
            print(f"Warning: Fixation map not found for {img_name}")
    
    pairs.sort(key=lambda x: os.path.basename(x[0]))
    if len(pairs) == 0:
        print("No valid image-fixation map pairs found for MIT1003.")
        return [], []
    
    test_image_list = []
    test_fixation_list = []
    for img_path, fix_path in pairs:
        dst_img = os.path.join(test_img_dir, os.path.basename(img_path))
        dst_fix = os.path.join(test_fix_dir, os.path.basename(fix_path))
        shutil.copy2(img_path, dst_img)
        shutil.copy2(fix_path, dst_fix)
        test_image_list.append(dst_img)
        test_fixation_list.append(dst_fix)
    
    print(f"MIT1003 test set: {len(test_image_list)} images, {len(test_fixation_list)} fixation maps")
    return test_image_list, test_fixation_list

def cleanup_mit1003():
    """Remove downloaded MIT1003 ZIP files and temporary extraction directories."""
    remove_file("ALLSTIMULI.zip")
    remove_file("ALLFIXATIONMAPS.zip")
    remove_tree("MIT1003_Stimuli")
    remove_tree("MIT1003_FixMaps")

###################################################
# Toronto dataset: Download, unify as test set, and cleanup functions
###################################################
def download_toronto():
    """Download the Toronto dataset if not already available."""
    toronto_url = "https://www-sop.inria.fr/members/Neil.Bruce/eyetrackingdata.zip"
    toronto_zip = norm_path("TorontoData.zip")
    toronto_dir = ensure_dir("Toronto_Raw")
    prepare_dataset(toronto_url, toronto_zip, toronto_dir)

def unify_toronto():
    """
    Unify the Toronto dataset as a test set.
    Organize images and corresponding fixation maps into data/Toronto/test.
    """
    base_data = ensure_dir("data")
    toronto_dir = ensure_dir(os.path.join(base_data, "Toronto"))
    test_img_dir = ensure_dir(os.path.join(toronto_dir, "test", "images"))
    test_fix_dir = ensure_dir(os.path.join(toronto_dir, "test", "fixmaps"))
    
    raw_root = norm_path("Toronto_Raw/eyetrackingdata/fixdens")
    img_root = norm_path(os.path.join(raw_root, "Original Image Set"))
    fix_root = norm_path(os.path.join(raw_root, "Density Maps produced from raw experimental eye tracking data"))
    
    all_images = sorted(glob.glob(os.path.join(img_root, "*.jpg")))
    test_image_list = []
    test_fixation_list = []
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        fix_name = "d" + img_name  # Fixation map filename expected to start with 'd'
        fix_path = os.path.join(fix_root, fix_name)
        if os.path.exists(fix_path):
            dst_img = os.path.join(test_img_dir, img_name)
            dst_fix = os.path.join(test_fix_dir, fix_name)
            shutil.copy2(img_path, dst_img)
            shutil.copy2(fix_path, dst_fix)
            test_image_list.append(dst_img)
            test_fixation_list.append(dst_fix)
        else:
            print(f"Warning: No fixation map for {img_name} (expected: {fix_name})")
    print(f"Toronto test set: {len(test_image_list)} images, {len(test_fixation_list)} fixation maps")
    return test_image_list, test_fixation_list

def cleanup_toronto():
    """Remove the downloaded Toronto ZIP file and temporary extraction directory."""
    remove_file("TorontoData.zip")
    remove_tree("Toronto_Raw")

###################################################
# SALICON dataset: Optimized processing using the SALICON API
###################################################
def download_salicon_data():
    """
    Download SALICON training and validation images along with corresponding annotation JSON files.
    Returns:
      train_extract_dir, val_extract_dir, train_ann_file, val_ann_file
    """
    # Download training images
    train_img_url = "https://s3.amazonaws.com/salicon-dataset/2015r1/train.zip"
    train_zip = norm_path("Salicon_train.zip")
    train_extract_dir = ensure_dir("SALICON_Train/train")
    prepare_dataset(train_img_url, train_zip, train_extract_dir)

    # Download validation images
    val_img_url = "https://s3.amazonaws.com/salicon-dataset/2015r1/val.zip"
    val_zip = norm_path("Salicon_val.zip")
    val_extract_dir = ensure_dir("SALICON_Val/val")
    prepare_dataset(val_img_url, val_zip, val_extract_dir)

    # Download annotation JSON files
    train_ann_url = "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_train2014.json"
    train_ann_file = norm_path("fixations_train2014.json")
    download_file(train_ann_url, train_ann_file)

    val_ann_url = "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_val2014.json"
    val_ann_file = norm_path("fixations_val2014.json")
    download_file(val_ann_url, val_ann_file)

    return train_extract_dir, val_extract_dir, train_ann_file, val_ann_file

def build_fixmap(fixations, width, height, sigma=20):
    """
    Build a fixation density map (fixmap) from a list of fixation points.
    Each fixation is provided as a [row, col] coordinate (1-indexed).
    Converts them to 0-indexed, accumulates the points, applies Gaussian smoothing,
    and normalizes the map to [0, 1].
    """
    fix_map = np.zeros((height, width), dtype=np.float32)
    for point in fixations:
        r = int(point[0]) - 1  # convert to 0-index
        c = int(point[1]) - 1
        if 0 <= r < height and 0 <= c < width:
            fix_map[r, c] += 1.0
    ksize = int(6 * sigma + 1)
    fix_map = cv2.GaussianBlur(fix_map, (ksize, ksize), sigma)
    if fix_map.max() > 0:
        fix_map = fix_map / fix_map.max()
    return fix_map

def unify_salicon_api(split, img_dir, annFile):
    """
    Unify the SALICON dataset for a given split ('train' or 'val') using the SALICON API.
    Organize images and generated fixation maps into data/SALICON/{split}.
    """
    sal = SALICON(annFile)
    imgIds = sal.getImgIds()
    
    base_data = ensure_dir("data")
    salicon_dir = ensure_dir(os.path.join(base_data, "SALICON"))
    dest_img_dir = ensure_dir(os.path.join(salicon_dir, split, "images"))
    dest_fix_dir = ensure_dir(os.path.join(salicon_dir, split, "fixmaps"))
    
    final_img_list = []
    final_fix_list = []
    for img_id in imgIds:
        img_info = sal.loadImgs([img_id])[0]
        src_img_path = os.path.join(img_dir, split, img_info["file_name"])
        if not os.path.exists(src_img_path):
            print(f"Warning: {src_img_path} not found.")
            continue
        img = cv2.imread(src_img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        annIds = sal.getAnnIds(imgIds=img_id)
        anns = sal.loadAnns(annIds)
        fixations = []
        for ann in anns:
            fixations.extend(ann.get("fixations", []))
        if len(fixations) == 0:
            print(f"No fixations for image {img_info['file_name']}")
            continue
        fix_map = build_fixmap(fixations, width, height, sigma=20)
        fix_map_uint8 = (fix_map * 255).astype(np.uint8)
        
        dst_img_path = os.path.join(dest_img_dir, img_info["file_name"])
        dst_fix_path = os.path.join(dest_fix_dir, os.path.splitext(img_info["file_name"])[0] + "_fixmap.png")
        shutil.copy2(src_img_path, dst_img_path)
        cv2.imwrite(dst_fix_path, fix_map_uint8)
        final_img_list.append(dst_img_path)
        final_fix_list.append(dst_fix_path)
    print(f"SALICON {split} set: {len(final_img_list)} images, {len(final_fix_list)} fixation maps")
    return final_img_list, final_fix_list

def cleanup_salicon():
    """Remove downloaded SALICON ZIP files, annotation JSON files, and temporary extraction directories."""
    remove_file("Salicon_train.zip")
    remove_file("Salicon_val.zip")
    remove_file("fixations_train2014.json")
    remove_file("fixations_val2014.json")
    remove_tree("SALICON_Train")
    remove_tree("SALICON_Val")

###################################################
# SaliencyDataset: PyTorch Dataset for loading image-fixation map pairs
###################################################
class SaliencyDatasetFromList(Dataset):
    def __init__(self, image_list, fixation_list, aug=None, target_size=224, retina_config="retinaParams.xml"):
        if len(image_list) != len(fixation_list):
            raise ValueError("Image and fixation list mismatch!")
        self.image_list = image_list
        self.fixation_list = fixation_list
        self.aug = aug
        self.target_size = target_size
        self.retina = cv2.bioinspired.Retina.create((self.target_size, self.target_size))
        self.retina.setup(retina_config)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        fix_path = self.fixation_list[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Cannot read image: " + img_path)

        img = cv2.resize(img, (self.target_size, self.target_size))
        self.retina.run(img)
        retina_img = self.retina.getParvo().astype(np.float32)

        sal = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
        sal = cv2.resize(sal, (self.target_size, self.target_size))
        if sal is None:
            raise ValueError("Cannot read fixation map: " + fix_path)

        if self.aug is not None:
            augmented = self.aug(image=retina_img, mask=sal)
            retina_img = augmented['image']
            sal = augmented['mask'].astype(np.float32) 
        
        sal = sal / sal.max()
        
        return np.transpose(retina_img, axes=[2, 0, 1]), np.expand_dims(sal, axis=0)


###################################################
# Helper function: Build file lists from given image and fixation map directories
###################################################
def get_file_lists(image_dir, fixmap_dir, img_exts=("*.jpg", "*.jpeg", "*.png")):
    image_list = []
    fixation_list = []
    for ext in img_exts:
        image_list.extend(glob.glob(os.path.join(image_dir, ext)))
        fixation_list.extend(glob.glob(os.path.join(fixmap_dir, ext)))
    image_list = sorted(image_list)
    fixation_list = sorted(fixation_list)
    return image_list, fixation_list

###################################################
# Main function: Prepare all datasets (MIT1003, Toronto, SALICON)
###################################################
def main():
    base_data = ensure_dir("data")
    retina = DummyRetina()  # Replace with your actual retina implementation
    
    # Process MIT1003 dataset (test set)
    mit_img_dir = os.path.join(base_data, "MIT1003", "test", "images")
    mit_fix_dir = os.path.join(base_data, "MIT1003", "test", "fixmaps")
    if not os.path.exists(mit_img_dir) or not os.listdir(mit_img_dir):
        print("Processing MIT1003 dataset...")
        download_mit1003()
        mit_image_list, mit_fixation_list = unify_mit1003_as_test()
        cleanup_mit1003()
    else:
        print("MIT1003 dataset already exists. Constructing SaliencyDatasetFromList...")
        mit_image_list, mit_fixation_list = get_file_lists(mit_img_dir, mit_fix_dir)
    
    mit_dataset = SaliencyDatasetFromList(mit_image_list, mit_fixation_list, retina)
    print(f"MIT1003 dataset contains {len(mit_dataset)} samples.")

    # Process Toronto dataset (test set)
    toronto_img_dir = os.path.join(base_data, "Toronto", "test", "images")
    toronto_fix_dir = os.path.join(base_data, "Toronto", "test", "fixmaps")
    if not os.path.exists(toronto_img_dir) or not os.listdir(toronto_img_dir):
        print("\nProcessing Toronto dataset...")
        toronto_image_list, toronto_fixation_list = unify_toronto()
        cleanup_toronto()
    else:
        print("Toronto dataset already exists. Constructing SaliencyDatasetFromList...")
        toronto_image_list, toronto_fixation_list = get_file_lists(toronto_img_dir, toronto_fix_dir)
    
    toronto_dataset = SaliencyDatasetFromList(toronto_image_list, toronto_fixation_list, retina)
    print(f"Toronto dataset contains {len(toronto_dataset)} samples.")

    # Process SALICON dataset (train and validation sets)
    sal_train_img_dir = os.path.join(base_data, "SALICON", "train", "images")
    sal_train_fix_dir = os.path.join(base_data, "SALICON", "train", "fixmaps")
    sal_val_img_dir = os.path.join(base_data, "SALICON", "val", "images")
    sal_val_fix_dir = os.path.join(base_data, "SALICON", "val", "fixmaps")
    
    if not os.path.exists(sal_train_img_dir) or not os.listdir(sal_train_img_dir):
        print("\nProcessing SALICON dataset...")
        train_img_dir, val_img_dir, train_ann_file, val_ann_file = download_salicon_data()
        sal_train_image_list, sal_train_fixation_list = unify_salicon_api("train", train_img_dir, train_ann_file)
        sal_val_image_list, sal_val_fixation_list = unify_salicon_api("val", val_img_dir, val_ann_file)
        cleanup_salicon()
    else:
        print("SALICON dataset already exists. Constructing SaliencyDatasetFromList for train and val...")
        sal_train_image_list, sal_train_fixation_list = get_file_lists(sal_train_img_dir, sal_train_fix_dir)
        sal_val_image_list, sal_val_fixation_list = get_file_lists(sal_val_img_dir, sal_val_fix_dir)
    
    salicon_train_dataset = SaliencyDatasetFromList(sal_train_image_list, sal_train_fixation_list, retina)
    salicon_val_dataset = SaliencyDatasetFromList(sal_val_image_list, sal_val_fixation_list, retina)
    print(f"SALICON train dataset contains {len(salicon_train_dataset)} samples.")
    print(f"SALICON val dataset contains {len(salicon_val_dataset)} samples.")

    print("\n--- Data Preparation Completed ---")

if __name__ == "__main__":
    main()
