import torch
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 criterion=None, optimizer=None, device='cpu', log_dir='./logs',
                 seed=42, early_stopping=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.seed = seed
        self.early_stopping = early_stopping
        self.best_val_loss = float('inf')
        self.writer = SummaryWriter(log_dir=log_dir)

    def fit(self, num_epochs, checkpoint_path):
        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = self._train_one_epoch()
            val_loss, val_acc, val_f1 = self._validate_one_epoch()

            self.writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch)
            self.writer.add_scalars('F1 Score', {'Train': train_f1, 'Validation': val_f1}, epoch)

            print(f'Epoch [{epoch + 1}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

            if self.early_stopping is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), checkpoint_path)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct_preds / total_preds
        f1_score = self._calculate_f1(correct_preds, total_preds)
        return avg_loss, accuracy, f1_score

    def _validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct_preds / total_preds
        f1_score = self._calculate_f1(correct_preds, total_preds)
        return avg_loss, accuracy, f1_score

    def _calculate_f1(self, correct_preds, total_preds):
        # Calcul simplifié du F1 Score
        precision = correct_preds / total_preds if total_preds > 0 else 0
        recall = correct_preds / total_preds if total_preds > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
