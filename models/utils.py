import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, root_dir, outcomes_csv, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.outcomes = pd.read_csv(outcomes_csv).set_index('PatientID')['Responder'].to_dict()
        self.patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_folder = os.path.join(self.root_dir, patient_id)

        pet_scan = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_folder, 'PET_lung.nrrd')))
        ct_scan = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_folder, 'CT_lung.nrrd')))
        merged_tumor = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_folder, 'merged_labeled_tumor_lung.nrrd')))

        lung_scan = np.stack([pet_scan, ct_scan], axis=0).astype(np.float32)
        lung_scan = torch.from_numpy(lung_scan)
        merged_tumor = torch.from_numpy(merged_tumor.astype(np.float32))

        if self.transforms:
            lung_scan = self.transforms(lung_scan)
            merged_tumor = self.transforms(merged_tumor)

        label = self.outcomes.get(patient_id, 0)
        label = torch.tensor(label, dtype=torch.float)

        return lung_scan, merged_tumor, label

def custom_collate_fn(batch):
    all_lung_scans = []
    all_merged_tumors = []
    all_labels = []

    for lung_scan, merged_tumor, label in batch:
        all_lung_scans.append(lung_scan)
        all_merged_tumors.append(merged_tumor)
        all_labels.append(label)
    
    lung_scans_tensor = torch.stack(all_lung_scans)
    merged_tumors_tensor = torch.stack(all_merged_tumors)
    labels_tensor = torch.tensor(all_labels)
    
    return lung_scans_tensor, merged_tumors_tensor, labels_tensor


class EarlyStopping:
    def __init__(self, patience=50, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, model_path='checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.66f}).  Saving model ...')
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for lung_scan, merged_tumor, labels in val_loader:
            lung_scan, merged_tumor, labels = lung_scan.to(device), merged_tumor.to(device), labels.to(device)
            outputs = model(lung_scan)
            preds = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_preds = np.array(all_preds) > 0.5  # Binarize predictions
    roc_auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    return roc_auc, accuracy, f1, precision, recall

def save_trial_info(trial, train_loss, val_loss, roc_auc, accuracy, f1, precision, recall):
    trial_info = {
        'trial_number': trial.number,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'batch_size': trial.params['batch_size'],
        'learning_rate': trial.params['lr'],
        'dropout_rate': trial.params['dropout_rate']
    }
    df = pd.DataFrame([trial_info])
    if not os.path.isfile('trial_info.csv'):
        df.to_csv('trial_info.csv', index=False)
    else:
        df.to_csv('trial_info.csv', mode='a', header=False, index=False)

import os
import torch
import nrrd
from torch.utils.data import Dataset
import pandas as pd

class CustomTumorDataset(Dataset):
    def __init__(self, root_dir, outcomes_csv, transform=None):
        self.root_dir = root_dir
        self.outcomes = pd.read_csv(outcomes_csv)
        self.transform = transform

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        patient_id = self.outcomes.iloc[idx, 0]
        pet_path = os.path.join(self.root_dir, patient_id, 'PET_tumor_lung.nrrd')
        ct_path = os.path.join(self.root_dir, patient_id, 'CT_tumor_lung.nrrd')
        seg_path = os.path.join(self.root_dir, patient_id, 'merged_labeled_tumor_lung.nrrd')

        pet_img, _ = nrrd.read(pet_path)
        ct_img, _ = nrrd.read(ct_path)
        seg_img, _ = nrrd.read(seg_path)
        
        # Normalize PET and CT scans
        pet_min, pet_max = pet_img.min(), pet_img.max()
        if pet_max - pet_min > 0:
            pet_img = (pet_img - pet_min) / (pet_max - pet_min) * 10.0
        else:
            pet_img = np.zeros_like(pet_img)  # or another appropriate default value

        ct_img = (ct_img - (-1000)) / (1000 - (-1000))  # Assuming CT values are in the range [-1000, 1000]

        pet_img = torch.tensor(pet_img, dtype=torch.float32).unsqueeze(0)
        ct_img = torch.tensor(ct_img, dtype=torch.float32).unsqueeze(0)
        seg_img = torch.tensor(seg_img, dtype=torch.int64)

        tumors = []
        tumor_labels = []

        min_size = (16, 32, 32)  # Minimum size to avoid shrinking issues
        for tumor_id in range(1, seg_img.max().item() + 1):
            tumor_mask = (seg_img == tumor_id).float()
            tumor_pet = pet_img * tumor_mask
            tumor_ct = ct_img * tumor_mask
            tumor_img = torch.cat((tumor_pet, tumor_ct), dim=0)
            # Resize tumor_img if necessary to match expected input size
            if tumor_img.shape[1:] < min_size:
                tumor_img = F.interpolate(tumor_img.unsqueeze(0), size=min_size, mode='trilinear', align_corners=False).squeeze(0)
            tumors.append(tumor_img)
            tumor_labels.append(self.outcomes.iloc[idx, 1])

        if self.transform:
            tumors = [self.transform(tumor) for tumor in tumors]
        
        return tumors, torch.tensor(tumor_labels, dtype=torch.float32)
