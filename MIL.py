## Usage sample: python MIL.py --epochs 10 --k_folds 5 --batch_size 10 --lr 0.001 --device cuda:4 --MIL_pooling mean --save_embeddings true
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from glob import glob
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/mnt/Dept_MachineLearning/Faculty/Rasool, Ghulam/Shared Resources/HNC-Histopath-Embeddings/matched_patients/', help="root dir")
    parser.add_argument('--csv_file', default='/mnt/Dept_MachineLearning/Faculty/Rasool, Ghulam/Shared Resources/HNC-Histopath-Embeddings/matched_patients/unique-labels.csv', help="label file")
    parser.add_argument('--results_dir', default='./MIL_results', help="results directory")
    parser.add_argument('--epochs', type=int, default=2, help="number of epochs")
    parser.add_argument('--k_folds', type=int, default=10, help="number of folds")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--device', type=str, default="cuda:4", help="device, cpu or cuda:0, cuda:1, ...")
    parser.add_argument('--MIL_pooling', type=str, default="attention", help="max, mean, attention")
    parser.add_argument('--save_embeddings', type=bool, default=False, help="save embeddings")
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    return opt

def print_options(parser, opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

opt = parse_args()
# Hyperparameters
epochs = opt.epochs
k_folds = opt.k_folds
batch_size = opt.batch_size
best_validation_accuracy = 0 

# Set the device
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Create a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Create the directory if it does not exist
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

# Attention Pooling Layer
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(AttentionPooling, self).__init__()
        self.attention_fc = nn.Linear(input_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, num_patches, num_features = x.size(0), x.size(1), x.size(2)
        x_flat = x.view(-1, num_features)
        attention_scores = F.relu(self.attention_fc(x_flat))
        attention_scores = torch.sigmoid(self.attention_weights(attention_scores))
        attention_scores = attention_scores.view(batch_size, num_patches, 1)
        attention_scores = F.softmax(attention_scores, dim=1)
        x_weighted = x * attention_scores
        x_aggregated = torch.sum(x_weighted, dim=1)
        return x_aggregated, attention_scores

# Attention MIL Model    
class AttnMIL(nn.Module):
    def __init__(self, MIL_pooling='attention'):
        super(AttnMIL, self).__init__()
        self.MIL_pooling = MIL_pooling
        self.fc1 = nn.Linear(100352, 12544)
        self.fc2 = nn.Linear(12544, 500)
        self.attention_pooling = AttentionPooling(500)
        self.embedding = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size, num_patches, num_features = x.shape
        x = x.view(-1, num_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(batch_size, num_patches, -1)
        if self.MIL_pooling == 'max':
            x = torch.max(x, dim=1).values
            attention_scores = None
        elif self.MIL_pooling == 'mean':
            x = torch.mean(x, dim=1)
            attention_scores = None
        else:
            x, attention_scores = self.attention_pooling(x)
        bag_embedding = F.relu(self.embedding(x))
        x = torch.sigmoid(self.fc3(bag_embedding))
        return x, bag_embedding, attention_scores

# Custom Dataloader
class PatientPatchDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with patient labels.
            root_dir (string): Directory with all the patch files.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.patient_ids = self.labels_frame['patient_id'].unique()

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_label = self.labels_frame[self.labels_frame['patient_id'] == patient_id]['label'].values[0]
        patch_files = glob(os.path.join(self.root_dir, f'{patient_id}*.npy'))
        all_patches = []
        for file in patch_files:
            patches_in_file = np.load(file)
            patches_in_file = patches_in_file.reshape(patches_in_file.shape[0], -1)
            all_patches.append(patches_in_file)
        if all_patches:
            all_patches = np.concatenate(all_patches, axis=0)
        else:
            all_patches = np.array([]).reshape(0, 100352)
        # sample = {'patches': all_patches, 'label': patient_label}
        sample = {'patient_id': patient_id, 'patches': all_patches, 'label': patient_label}
        return sample

# Pad the patches to make them of equal length
def custom_collate(batch):
    patches = [torch.tensor(item['patches']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    patches_padded = pad_sequence(patches, batch_first=True, padding_value=0)
    return {'patches': patches_padded, 'label': labels}

# Training and Validation Functions
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    loss_epoch, correct_predictions, total_samples, acc_epoch = 0, 0, 0, 0
    for batch in train_loader:
        inputs, labels = batch['patches'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(inputs)
        outputs = outputs.view(-1)
        labels = labels.view(-1)
        predicted = outputs.sigmoid().round() if outputs.dim() == 2 else outputs.round()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    loss_epoch /= len(train_loader)
    acc_epoch = 100. * correct_predictions / total_samples
    return loss_epoch, acc_epoch

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss, correct_preds, total_samples, val_acc = 0, 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['patches'].to(device), batch['label'].to(device)
            outputs, _, _ = model(inputs)
            outputs = outputs.view(-1)
            preds = outputs.sigmoid().round() if outputs.dim() == 2 else outputs.round()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            correct_preds += preds.eq(labels.view_as(preds)).sum().item()
            total_samples += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    val_loss /= len(val_loader)
    val_acc = 100. * correct_preds / total_samples
    return val_loss, val_acc, y_true, y_pred

# K-Fold Cross Validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize lists to store the results
results_train_loss, results_train_acc = [], []
results_val_loss, results_val_acc = [], []
train_y_trues, train_y_preds = [], []
val_y_trues, val_y_preds = [], []

# Load the dataset
dataset = PatientPatchDataset(csv_file=opt.csv_file, root_dir=opt.root_dir)
print('Length of Dataset', len(dataset))

# Address class imbalance
class_0 = len(dataset.labels_frame[dataset.labels_frame['label'] == 0])
class_1 = len(dataset.labels_frame[dataset.labels_frame['label'] == 1])
pos_weight = torch.tensor([class_0 / class_1]).to(device)
print('Class 0:', class_0, ', Class 1:', class_1)
print('Positive class weight:', pos_weight)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Train the model
for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset)))):
    print(f"Fold {fold + 1}/{k_folds}")
    train_subsampler = Subset(dataset, train_idx)
    val_subsampler = Subset(dataset, test_idx)
    train_loader = DataLoader(dataset=train_subsampler, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=val_subsampler, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=8, pin_memory=True)
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2024)
    torch.manual_seed(2024)
    random.seed(2024)
    model = AttnMIL(MIL_pooling=opt.MIL_pooling).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    for epoch in range(epochs):
        loss_epoch, acc_epoch = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch+1}, Train Epoch Loss: {loss_epoch:.3f}, Train Accuracy: {acc_epoch:.2f}%")
        # if epoch == 1:
        #     break
    fold_train_loss, fold_train_acc, train_y_true, train_y_pred = validate(model, device, train_loader, criterion)
    fold_val_loss, fold_val_acc, val_y_true, val_y_pred = validate(model, device, val_loader, criterion)
    print(f"Fold {fold+1}, Fold Train Loss: {fold_train_loss:.3f}, Fold Train Accuracy: {fold_train_acc:.2f}%")
    print(f"Fold {fold+1}, Fold Validation Loss: {fold_val_loss:.3f}, Fold Validation Accuracy: {fold_val_acc:.2f}%")

    # Save the model if it has the best validation accuracy so far
    if fold_val_acc > best_validation_accuracy:
        best_validation_accuracy = fold_val_acc
        torch.save(model.state_dict(), 'best_MIL_model.pth')

    results_train_loss.append(round(fold_train_loss, 2))
    results_train_acc.append(round(fold_train_acc, 2))
    results_val_loss.append(round(fold_val_loss, 2)) 
    results_val_acc.append(round(fold_val_acc, 2))
    train_y_trues.extend(train_y_true)
    train_y_preds.extend(train_y_pred)
    val_y_trues.extend(val_y_true)
    val_y_preds.extend(val_y_pred)
    class_names = ['Negative', 'Positive']
    plot_confusion_matrix(train_y_trues, train_y_preds, classes=class_names, title='Confusion matrix')
    plot_confusion_matrix(val_y_trues, val_y_preds, classes=class_names, title='Confusion matrix')
    plt.show()
    
    # save the confusion matrix figure
    plt.savefig(f'./MIL_results/confusion_matrix_fold_{fold+1}.png')
    plt.close()

print()
print(f'Folds Training Loss: {results_train_loss}')
print(f'Folds Validation Loss: {results_val_loss}')
print(f'Folds Training Accuracy: {results_train_acc}')
print(f'Folds Validation Accuracy: {results_val_acc}')
print(f'Average Training Loss: {np.array(results_train_loss).mean():.3f}, Standard Deviation: {np.array(results_train_loss).std():.3f}')
print(f'Average Validation Loss: {np.array(results_val_loss).mean():.3f}, Standard Deviation: {np.array(results_val_loss).std():.3f}')
print(f'Average Training Accuracy: {np.array(results_train_acc).mean():.2f}, Standard Deviation: {np.array(results_train_acc).std():.2f}')
print(f'Average Validation Accuracy: {np.array(results_val_acc).mean():.2f}, Standard Deviation: {np.array(results_val_acc).std():.2f}')

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, k_folds + 1), results_train_acc, label='Training Accuracy')
plt.plot(np.arange(1, k_folds + 1), results_val_acc, label='Validation Accuracy')
plt.xticks(np.arange(1, k_folds + 1))
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
# save the figure
plt.savefig('./MIL_results/accuracy.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, k_folds + 1), results_train_loss, label='Training Loss')
plt.plot(np.arange(1, k_folds + 1), results_val_loss, label='Validation Loss')
plt.xticks(np.arange(1, k_folds + 1))
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# save the figure
plt.savefig('./MIL_results/loss.png')
plt.close()

# Save the embeddings
if opt.save_embeddings:
    print('Saving embeddings')
    print('---> For individual embedding files, setting Batch_size=1')
    model = AttnMIL(MIL_pooling=opt.MIL_pooling).to(device)
    model.load_state_dict(torch.load('best_MIL_model.pth'))
    model.eval()
    dataset = PatientPatchDataset(csv_file=opt.csv_file, root_dir=opt.root_dir)
    # get the patient ids
    patients = []
    patients.append(dataset.patient_ids)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=8, pin_memory=True)
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, label = batch['patches'].to(device), batch['label'].to(device)
            _, bag_embedding, _ = model(inputs)
            embeddings.append(bag_embedding.cpu().numpy())
            labels.append(label.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    np.save(f'{opt.results_dir}/embeddings.npy', embeddings)
    np.save(f'{opt.results_dir}/labels.npy', labels)
    np.save(f'{opt.results_dir}/patients.npy', patients)
    print('Embeddings saved')
    print('Labels saved')
    print('Patient IDs saved')
    print('Embeddings shape:', embeddings.shape)
    print('Labels shape:', labels.shape)