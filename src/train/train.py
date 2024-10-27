import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import numpy as np

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train_model(model, dataloaders, epochs, learning_rate, log_path, save_model_path, device):
    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        print(f"Training on fold {fold + 1}/{len(dataloaders)}")

        model.apply(reset_weights)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = BCEWithLogitsLoss()

        best_val_acc = 0.0
        best_epoch = 0
        best_model_path = None

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_train_outputs = []
            all_train_labels = []

            progress_bar = tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1}/{epochs}", leave=False)

            for text_data, audio_data, label in progress_bar:
                optimizer.zero_grad()

                text_data = {key: value.squeeze(1).to(device) for key, value in text_data.items()}
                audio_data = audio_data.to(device)
                label = label.to(device)

                outputs = model(text_data, audio_data)
                loss = criterion(outputs, label.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                all_train_outputs.extend(outputs.detach().cpu().numpy())
                all_train_labels.extend(label.cpu().numpy())

                progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1)})

            
            train_predictions = (np.array(all_train_outputs) > 0.5).astype(int)
            train_report = classification_report(np.array(all_train_labels), train_predictions, zero_division=0)


            val_loss, val_acc, val_report = validate_model(model, val_loader, criterion, device)

            with open(log_path, 'a') as f:
                f.write(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {running_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}\n")
                f.write(f"Training Classification Report (Fold {fold+1}, Epoch {epoch+1}):\n{train_report}\n")
                f.write(f"Validation Classification Report (Fold {fold+1}, Epoch {epoch+1}):\n{val_report}\n")

            print(f"Training Classification Report (Epoch {epoch+1}):\n{train_report}")
            print(f"Validation Classification Report (Epoch {epoch+1}):\n{val_report}")


            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1

                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                model_save_path = f"{save_model_path}_fold{fold+1}_epoch{best_epoch}.pth"

                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)

                best_model_path = model_save_path

                print(f"Best model saved for fold {fold+1} at epoch {best_epoch} with val_acc {best_val_acc:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"Best model for fold {fold+1} saved at epoch {best_epoch} with validation accuracy {best_val_acc:.4f}\n")


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for text_data, audio_data, label in val_loader:
            text_data = {key: value.squeeze(1).to(device) for key, value in text_data.items()}
            audio_data = audio_data.to(device)
            label = label.to(device)

            outputs = model(text_data, audio_data)
            loss = criterion(outputs, label.unsqueeze(1).float())
            val_loss += loss.item()

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    predictions = (np.array(all_outputs) > 0.5).astype(int)
    accuracy = accuracy_score(np.array(all_labels), predictions)

    report = classification_report(np.array(all_labels), predictions, zero_division=0)
    
    return val_loss, accuracy, report



