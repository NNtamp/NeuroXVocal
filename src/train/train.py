import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm


import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def train_model(model, dataloaders, epochs, learning_rate, log_path, save_model_path, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()

    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        print(f"Training on fold {fold+1}/{len(dataloaders)}")
        
        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_train_outputs = []  
            all_train_labels = []  
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for text_data, audio_data, label in progress_bar:
                optimizer.zero_grad()

                text_data = {key: value.squeeze(1).to(device) for key, value in text_data.items()}
                audio_data = audio_data.to(device)
                label = label.to(device)

                outputs = model(text_data, audio_data)
                loss = criterion(outputs, label.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                all_train_outputs.extend(outputs.detach().cpu().numpy())
                all_train_labels.extend(label.cpu().numpy())

                progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1)})

            train_predictions = (np.array(all_train_outputs) > 0.5).astype(int)
            train_report = classification_report(np.array(all_train_labels), train_predictions)

            val_loss, val_acc, val_report = validate_model(model, val_loader, criterion, device)

            with open(log_path, 'a') as f:
                f.write(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {running_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}\n")
                f.write(f"Training Classification Report (Fold {fold+1}, Epoch {epoch+1}):\n{train_report}\n")
                f.write(f"Validation Classification Report (Fold {fold+1}, Epoch {epoch+1}):\n{val_report}\n")

            print(f"Training Classification Report (Epoch {epoch+1}):\n{train_report}")
            print(f"Validation Classification Report (Epoch {epoch+1}):\n{val_report}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{save_model_path}_best_fold{fold+1}.pth")
                print(f"Best model saved for fold {fold+1} at epoch {epoch+1}")


            
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
            loss = criterion(outputs, label.unsqueeze(1))
            val_loss += loss.item()

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    predictions = (np.array(all_outputs) > 0.5).astype(int)
    accuracy = accuracy_score(np.array(all_labels), predictions)

    report = classification_report(np.array(all_labels), predictions)
    
    return val_loss, accuracy, report



