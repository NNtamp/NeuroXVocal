import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import os
from config import BATCH_SIZE, EARLY_STOPPING_PATIENCE
import torch.nn as nn 

def train_model(
    model,
    full_dataset,
    epochs,
    learning_rate,
    log_path,
    save_model_path,
    device,
    num_folds=5,
    save_best_model=False
):
    criterion = BCEWithLogitsLoss()

    kfold = KFold(n_splits=num_folds, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_dataset)):
        print(f'Fold {fold+1}/{num_folds}')

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        if isinstance(model, nn.DataParallel):
            model.module.reset_parameters() 
        else:
            model.reset_parameters()

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_train_outputs = []
            all_train_labels = []

            for (
                text_data,
                audio_data,
                embedding_data,
                label
            ) in tqdm(
                train_loader,
                desc=f"Fold {fold+1}, Epoch {epoch+1}/{epochs} - Training"
            ):
                optimizer.zero_grad()
                text_data = {
                    key: value.to(device) for key, value in text_data.items()
                }
                audio_data = audio_data.to(device)
                embedding_data = embedding_data.to(device)
                label = label.to(device)

                outputs = model(text_data, audio_data, embedding_data)

                loss = criterion(outputs, label.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                all_train_outputs.extend(outputs.detach().cpu().numpy())
                all_train_labels.extend(label.cpu().numpy())

            train_loss = running_loss / len(train_loader)
            train_predictions = (np.array(all_train_outputs) > 0).astype(int)
            train_accuracy = accuracy_score(
                np.array(all_train_labels),
                train_predictions
            )
            train_report = classification_report(
                np.array(all_train_labels),
                train_predictions,
                target_names=['Control', 'ProbableAD'],
                digits=4
            )

            model.eval()
            val_running_loss = 0.0
            all_val_outputs = []
            all_val_labels = []
            with torch.no_grad():
                for (
                    text_data,
                    audio_data,
                    embedding_data,
                    label
                ) in tqdm(
                    val_loader,
                    desc=f"Fold {fold+1}, Epoch {epoch+1}/{epochs} - Validation"
                ):
                    text_data = {
                        key: value.to(device) for key, value in text_data.items()
                    }
                    audio_data = audio_data.to(device)
                    embedding_data = embedding_data.to(device)
                    label = label.to(device)

                    outputs = model(text_data, audio_data, embedding_data)
                    loss = criterion(outputs, label.float())

                    val_running_loss += loss.item()
                    all_val_outputs.extend(outputs.detach().cpu().numpy())
                    all_val_labels.extend(label.cpu().numpy())

            val_loss = val_running_loss / len(val_loader)
            val_predictions = (np.array(all_val_outputs) > 0).astype(int)
            val_accuracy = accuracy_score(
                np.array(all_val_labels),
                val_predictions
            )
            val_report = classification_report(
                np.array(all_val_labels),
                val_predictions,
                target_names=['Control', 'ProbableAD'],
                digits=4
            )

            scheduler.step(val_loss)


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if save_best_model:
                    best_model_path = f"{save_model_path}_fold{fold+1}_best.pth"
                    if isinstance(model, torch.nn.DataParallel):
                        torch.save(model.module.state_dict(), best_model_path)
                    else:
                        torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                    break

            with open(log_path, 'a') as f:
                f.write(
                    f"Fold {fold+1}, Epoch {epoch+1}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                )
                f.write("Train Classification Report:\n")
                f.write(f"{train_report}\n")
                f.write("Validation Classification Report:\n")
                f.write(f"{val_report}\n")

            print(
                f"Fold {fold+1}, Epoch {epoch+1}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            print("Train Classification Report:")
            print(train_report)
            print("Validation Classification Report:")
            print(val_report)
