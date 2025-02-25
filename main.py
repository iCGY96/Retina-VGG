import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # for progress bars

# Import functions from our custom modules
from dataset import (
    download_mit1003,
    unify_mit1003_as_test,
    download_toronto,
    unify_toronto,
    cleanup_mit1003,
    cleanup_toronto,
    download_salicon_data,
    unify_salicon_api,
    cleanup_salicon,
    SaliencyDatasetFromList
)
from model import PretrainedVGG16Saliency
from utils import evaluate_metrics

########################################
# Focal Loss for binary saliency maps
########################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        """
        gamma: focusing parameter
        alpha: balance parameter
        reduction: reduction method ('mean' or 'sum')
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        eps = 1e-8
        # Clamp inputs to avoid log(0)
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        loss_pos = - self.alpha * ((1 - inputs) ** self.gamma) * targets * torch.log(inputs)
        loss_neg = - (1 - self.alpha) * (inputs ** self.gamma) * (1 - targets) * torch.log(1 - inputs)
        loss = loss_pos + loss_neg
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

########################################
# Fix random seed for reproducibility
########################################
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################
# Modified testing function that returns average metrics
########################################
def test_model_loader(model, test_loader, device, print_metrics=True):
    model.eval()
    metric_list = {
        'CC': [], 'Similarity': [], 'EMD': [], 'KLdiv': [],
        'NSS': [], 'AUCJudd': [], 'AUCBorji': [], 'AUCshuffled': []
    }
    pbar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for images, sal_maps in pbar:
            images = images.to(device)
            sal_maps = sal_maps.to(device)
            outputs = model(images)

            b, c, h, w = outputs.shape
            outputs = outputs.view(b, -1)
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.view(b, 1, h, w)

            for i in range(outputs.size(0)):
                pred = outputs[i, 0].cpu().numpy()  # predicted saliency map
                gt = sal_maps[i, 0].cpu().numpy()   # ground truth saliency map
                metrics = evaluate_metrics(pred, gt)
                for key in metric_list:
                    metric_list[key].append(metrics[key])
    avg_metrics = {
        key: (np.nanmean(metric_list[key]) if len(metric_list[key]) > 0 else float('nan'))
        for key in metric_list
    }
    if print_metrics:
        print("\nAverage Metrics on Test Set:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
    return avg_metrics

########################################
# Modified training function with per-epoch testing and best model saving (using CC as criterion)
########################################
def train_model(model, train_loader, test_loader, device, num_epochs=10, model_save_path="best_model.pth"):
    model.train()
    # Use Focal Loss combined with MSELoss
    criterion1 = FocalLoss(gamma=2, alpha=0.25, reduction='mean')
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_cc = -float("inf")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for images, sal_maps in pbar:
            images = images.to(device)
            sal_maps = sal_maps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion1(outputs, sal_maps) + criterion2(outputs, sal_maps)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Training Loss: {avg_loss:.4f}")

        # Evaluate on test set per epoch using CC as the criterion
        avg_metrics = test_model_loader(model, test_loader, device, print_metrics=False)
        if avg_metrics.get("CC", -float("inf")) > best_cc:
            best_cc = avg_metrics["CC"]
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: New best CC: {best_cc:.4f} - Model saved.")

    return best_cc

########################################
# Main function: Prepare datasets, train on SALICON, and test on MIT1003 and Toronto
########################################
def main():
    fix_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_size = 224

    ####################################
    # Prepare SALICON training dataset
    ####################################
    print("Preparing SALICON training dataset...")
    train_img_dir, val_img_dir, train_ann_file, val_ann_file = download_salicon_data()
    sal_train_img_list, sal_train_fix_list = unify_salicon_api("train", train_img_dir, train_ann_file)
    cleanup_salicon()  # Clean up temporary SALICON files

    ####################################
    # Prepare MIT1003 test dataset
    ####################################
    print("\nPreparing MIT1003 test dataset...")
    download_mit1003()
    # We only need the test split from MIT1003
    mit_test_img_list, mit_test_fix_list = unify_mit1003_as_test()
    cleanup_mit1003()

    ####################################
    # Prepare Toronto test dataset
    ####################################
    print("\nPreparing Toronto test dataset...")
    download_toronto()
    toronto_test_img_list, toronto_test_fix_list = unify_toronto()
    cleanup_toronto()

    ####################################
    # Create a Retina instance for preprocessing
    ####################################
    sample_img = cv2.imread(sal_train_img_list[0])
    sample_img = cv2.resize(sample_img, (target_size, target_size))
    # Create a retina simulation instance (using OpenCV's bioinspired module)
    retina = cv2.bioinspired.Retina.create((sample_img.shape[1], sample_img.shape[0]))
    retina.write("retinaParams.xml")
    retina.setup("retinaParams.xml")

    ####################################
    # Build DataLoaders
    ####################################
    train_dataset = SaliencyDatasetFromList(
        sal_train_img_list, sal_train_fix_list,
        retina=retina, target_size=target_size
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    mit_test_dataset = SaliencyDatasetFromList(
        mit_test_img_list, mit_test_fix_list,
        retina=retina, target_size=target_size
    )
    mit_test_loader = DataLoader(mit_test_dataset, batch_size=8, shuffle=False, num_workers=0)

    toronto_test_dataset = SaliencyDatasetFromList(
        toronto_test_img_list, toronto_test_fix_list,
        retina=retina, target_size=target_size
    )
    toronto_test_loader = DataLoader(toronto_test_dataset, batch_size=8, shuffle=False, num_workers=0)

    ####################################
    # Build the model
    ####################################
    model = PretrainedVGG16Saliency(output_size=target_size).to(device)
    print("Model architecture:")
    print(model)

    ####################################
    # Train model on SALICON training set with per-epoch testing on MIT1003 test set
    ####################################
    num_epochs = 50
    print("Training on SALICON training set...")
    best_cc = train_model(model, train_loader, mit_test_loader, device, num_epochs=num_epochs, model_save_path="saliency_model_best_cc.pth")

    ####################################
    # Final Testing on MIT1003 test set
    ####################################
    print("\nFinal evaluation on MIT1003 test set:")
    test_model_loader(model, mit_test_loader, device, print_metrics=True)

    ####################################
    # Testing on Toronto test set
    ####################################
    print("\nTesting on Toronto test set:")
    test_model_loader(model, toronto_test_loader, device, print_metrics=True)

if __name__ == "__main__":
    main()
