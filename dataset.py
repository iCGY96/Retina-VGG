import os
import glob
import shutil
import requests
import zipfile
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from salicon.salicon import SALICON  # Use SALICON API for annotations

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
    """Download file from a URL to the destination path."""
    dest_path = norm_path(dest_path)
    if not os.path.exists(dest_path):
        print(f"Downloading from {url} to {dest_path} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed.")
    else:
        print(f"{dest_path} already exists.")

def extract_zip(zip_path: str, extract_to: str):
    """Extract a ZIP file to the given directory."""
    zip_path = norm_path(zip_path)
    extract_to = ensure_dir(extract_to)
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def prepare_dataset(url: str, zip_filename: str, extract_dir: str):
    """Download and extract dataset from a URL."""
    download_file(url, zip_filename)
    extract_zip(zip_filename, extract_dir)

###################################################
# MIT1003 dataset: Download, unify as test set, and cleanup functions
###################################################
def download_mit1003():
    """Download MIT1003 stimuli images and fixation maps."""
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
    Reads images from MIT1003_Stimuli/ALLSTIMULI and corresponding fixation maps from
    MIT1003_FixMaps/ALLFIXATIONMAPS (filename: original name + '_fixMap.jpg'),
    then copies valid pairs to data/MIT1003/test/images and data/MIT1003/test/fixmaps.
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
    """Download the Toronto dataset."""
    toronto_url = "https://www-sop.inria.fr/members/Neil.Bruce/eyetrackingdata.zip"
    toronto_zip = norm_path("TorontoData.zip")
    toronto_dir = ensure_dir("Toronto_Raw")
    prepare_dataset(toronto_url, toronto_zip, toronto_dir)

def unify_toronto():
    """
    Unify the Toronto dataset as a test set.
    Reads images from 'Toronto_Raw/eyetrackingdata/fixdens/Original Image Set' and corresponding
    fixation maps from 'Toronto_Raw/eyetrackingdata/fixdens/Density Maps produced from raw experimental eye tracking data',
    then copies valid pairs to data/Toronto/test/images and data/Toronto/test/fixmaps.
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
        fix_name = "d" + img_name  # Expected fixation map filename starts with 'd'
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
    The function converts them to 0-indexed, accumulates the points,
    applies Gaussian smoothing, and normalizes the map to [0, 1].
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
    Loads annotations with SALICON(annFile) and uses API functions to retrieve image info and fixations.
    Generates fixation maps with build_fixmap, and copies images and generated maps into
    data/SALICON/{split}/images and data/SALICON/{split}/fixmaps.
    Returns lists of image paths and corresponding fixation map paths.
    """
    # Initialize SALICON API
    sal = SALICON(annFile)
    imgIds = sal.getImgIds()
    
    base_data = ensure_dir("data")
    salicon_dir = ensure_dir(os.path.join(base_data, "SALICON"))
    dest_img_dir = ensure_dir(os.path.join(salicon_dir, split, "images"))
    dest_fix_dir = ensure_dir(os.path.join(salicon_dir, split, "fixmaps"))
    
    final_img_list = []
    final_fix_list = []
    for img_id in imgIds:
        # Use the SALICON API to load image info
        img_info = sal.loadImgs([img_id])[0]
        # Construct the source image path (assumes images are directly under img_dir)
        src_img_path = os.path.join(img_dir, split, img_info["file_name"])
        if not os.path.exists(src_img_path):
            print(f"Warning: {src_img_path} not found.")
            continue
        # Load image to obtain its dimensions
        img = cv2.imread(src_img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        # Retrieve all annotations (fixations) for this image using the API
        annIds = sal.getAnnIds(imgIds=img_id)
        anns = sal.loadAnns(annIds)
        fixations = []
        for ann in anns:
            fixations.extend(ann.get("fixations", []))
        if len(fixations) == 0:
            print(f"No fixations for image {img_info['file_name']}")
            continue
        # Generate the fixation map using build_fixmap
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
    def __init__(self, image_list, fixation_list, retina, target_size=224):
        if len(image_list) != len(fixation_list):
            raise ValueError("Image and fixation list mismatch!")
        self.image_list = image_list
        self.fixation_list = fixation_list
        self.retina = retina
        self.target_size = target_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.image_list[idx]
        fix_path = self.fixation_list[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Cannot read image: " + img_path)
        img = cv2.resize(img, (self.target_size, self.target_size))
        # Process image using retina simulation (assumed provided externally)
        self.retina.run(img)
        retina_img = self.retina.getParvo().astype(np.float32) / 255.0
        retina_img = np.transpose(retina_img, (2, 0, 1))
        # Load and preprocess fixation map (grayscale)
        sal = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
        if sal is None:
            raise ValueError("Cannot read fixation map: " + fix_path)
        sal = cv2.resize(sal, (self.target_size, self.target_size))
        sal = sal.astype(np.float32) / 255.0
        sal = np.expand_dims(sal, axis=0)
        return torch.from_numpy(retina_img), torch.from_numpy(sal)

###################################################
# Main function: Prepare all datasets (MIT1003, Toronto, SALICON)
###################################################
def main():
    # --- Process MIT1003 dataset as test set ---
    print("Processing MIT1003 dataset...")
    download_mit1003()
    mit_test_img_list, mit_test_fix_list = unify_mit1003_as_test()
    cleanup_mit1003()

    # --- Process Toronto dataset (test set) ---
    print("\nProcessing Toronto dataset...")
    download_toronto()
    toronto_test_img_list, toronto_test_fix_list = unify_toronto()
    cleanup_toronto()

    # --- Process SALICON dataset using SALICON API for training and validation sets ---
    print("\nProcessing SALICON dataset...")
    train_img_dir, val_img_dir, train_ann_file, val_ann_file = download_salicon_data()
    sal_train_img_list, sal_train_fix_list = unify_salicon_api("train", train_img_dir, train_ann_file)
    sal_val_img_list, sal_val_fix_list = unify_salicon_api("val", val_img_dir, val_ann_file)
    cleanup_salicon()

    print("\n--- Data Preparation Completed ---")
    print(f"MIT1003 Test: {len(mit_test_img_list)} images, {len(mit_test_fix_list)} fixation maps")
    print(f"Toronto Test:  {len(toronto_test_img_list)} images, {len(toronto_test_fix_list)} fixation maps")
    print(f"SALICON Train: {len(sal_train_img_list)} images, {len(sal_train_fix_list)} fixation maps")
    print(f"SALICON Val:   {len(sal_val_img_list)} images, {len(sal_val_fix_list)} fixation maps")

if __name__ == "__main__":
    main()
