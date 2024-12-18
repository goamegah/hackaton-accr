import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

import random
import os


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Fonction de calcul de la précision top-1 et top-5
def accuracy(output, target, topk=(1,)):
    """Calculer la précision top-k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.view(1, -1).expand_as(pred)
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Fonction pour calculer la matrice de confusion
def confusion_matrix_torch(y_true, y_pred, num_classes):
    """
    Calcule une matrice de confusion sans utiliser scikit-learn.
    
    Parameters:
        - y_true (torch.Tensor): Vérités de terrain.
        - y_pred (torch.Tensor): Prédictions du modèle.
        - num_classes (int): Le nombre de classes.
        
    Returns:
        - matrix (np.ndarray): Matrice de confusion.
    """
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    
    return matrix.numpy()


# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(matrix, class_names, title='Confusion Matrix'):
    """
    Affiche une matrice de confusion sous forme de heatmap.
    
    Parameters:
        - matrix (np.ndarray): Matrice de confusion.
        - class_names (list): Liste des noms des classes.
        - title (str): Titre de la figure.
    """
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Fonction pour calculer le F1 score
def f1_score_torch(y_true, y_pred, num_classes):
    """
    Calcule le F1-score sans utiliser scikit-learn.

    Parameters:
        - y_true (torch.Tensor): Vérités de terrain.
        - y_pred (torch.Tensor): Prédictions du modèle.
        - num_classes (int): Le nombre de classes.

    Returns:
        - f1 (float): Le score F1 moyen.
    """
    # Calcul de la matrice de confusion
    cm = confusion_matrix_torch(y_true, y_pred, num_classes)
    
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    
    # Calcul du F1 score pour chaque classe
    f1_per_class = 2 * (precision * recall) / (precision + recall)
    f1_per_class = np.nan_to_num(f1_per_class)  # Remplacer NaN par 0 pour les classes sans prédictions
    
    return np.mean(f1_per_class)  # Renvoie le F1-score moyen


# Fonction d'entraînement
def train(model, train_loader, criterion, optimizer, num_epochs, args, writer):
    """
    Entraînement du modèle avec logs dans TensorBoard et calcul des métriques.

    Parameters:
        - model (torch.nn.Module): Le modèle PyTorch à entraîner.
        - train_loader (DataLoader): Dataloader pour l'ensemble d'entraînement.
        - criterion (nn.Module): Fonction de perte.
        - optimizer (torch.optim.Optimizer): Optimiseur pour l'entraînement.
        - num_epochs (int): Nombre d'époques d'entraînement.
        - args (namespace): Arguments avec des informations comme 'device' et autres paramètres.
        - writer (SummaryWriter): Objet TensorBoardX pour l'enregistrement des logs.
    """
    for epoch in range(num_epochs):
        model.train()  # Mettre le modèle en mode entraînement
        epoch_loss = 0
        correct = 0
        total = 0
        y_true, y_pred = [], []

        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            # Zéro les gradients des optimisateurs
            optimizer.zero_grad()

            # Prédictions du modèle
            logits = model(x_batch)

            # Calcul de la perte
            loss = criterion(logits, y_batch)
            loss.backward()

            # Mise à jour des poids
            optimizer.step()

            # Accumuler les métriques
            epoch_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == y_batch).item()
            total += y_batch.size(0)

            # Stocker les valeurs pour la matrice de confusion
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        # Calculs des métriques
        epoch_loss /= (counter + 1)
        accuracy = 100 * correct / total
        f1 = f1_score_torch(torch.tensor(y_true), torch.tensor(y_pred), num_classes=len(np.unique(y_true)))
        print(f"Epoch {epoch + 1}/{num_epochs}\t"
              f"Loss: {epoch_loss:.4f}\t"
              f"Accuracy: {accuracy:.4f}%\t"
              f"F1-Score: {f1:.4f}")

        # Logs dans TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', accuracy, epoch)
        writer.add_scalar('F1-Score/Train', f1, epoch)


# Fonction d'évaluation
def eval(model, val_loader, criterion, args, writer, epoch):
    """
    Évaluation du modèle sur l'ensemble de validation avec logs dans TensorBoard 
    et affichage de la matrice de confusion.

    Parameters:
        - model (torch.nn.Module): Le modèle PyTorch à évaluer.
        - val_loader (DataLoader): Dataloader pour l'ensemble de validation.
        - criterion (nn.Module): Fonction de perte.
        - args (namespace): Arguments avec des informations comme 'device' et autres paramètres.
        - writer (SummaryWriter): Objet TensorBoardX pour l'enregistrement des logs.
        - epoch (int): L'indice de l'époque courante.
    """
    model.eval()  # Mettre le modèle en mode évaluation
    val_loss = 0
    val_correct = 0
    val_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():  # Pas de gradients pendant l'évaluation
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item()
            _, preds = torch.max(logits, 1)
            val_correct += torch.sum(preds == y_batch).item()
            val_total += y_batch.size(0)

            # Stocker les valeurs pour la matrice de confusion
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calcul des métriques de validation
    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    f1 = f1_score_torch(torch.tensor(y_true), torch.tensor(y_pred), num_classes=len(np.unique(y_true)))
    print(f"Validation Loss: {val_loss:.4f}\t"
          f"Validation Accuracy: {val_accuracy:.4f}%\t"
          f"Validation F1-Score: {f1:.4f}")

    # Logs dans TensorBoard pour validation
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
    writer.add_scalar('F1-Score/Val', f1, epoch)

    # Matrice de confusion sur la validation
    num_classes = len(np.unique(y_true))
    cm = confusion_matrix_torch(torch.tensor(y_true), torch.tensor(y_pred), num_classes)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(num_classes)], title=f'Confusion Matrix - Epoch {epoch + 1}')
    writer.add_figure(f"Confusion Matrix/Epoch_{epoch}", plot_confusion_matrix(cm, class_names=[str(i) for i in range(num_classes)]), global_step=epoch)


# Fonction d'entraînement complète avec évaluation
def train_and_eval(model, train_loader, val_loader, criterion, optimizer, num_epochs, args):
    """
    Fonction principale qui appelle les fonctions d'entraînement et d'évaluation, 
    et qui enregistre les logs dans TensorBoard.
    
    Parameters:
        - model (torch.nn.Module): Le modèle PyTorch à entraîner et à évaluer.
        - train_loader (DataLoader): Dataloader pour l'ensemble d'entraînement.
        - val_loader (DataLoader): Dataloader pour l'ensemble de validation.
        - criterion (nn.Module): Fonction de perte.
        - optimizer (torch.optim.Optimizer): Optimiseur pour l'entraînement.
        - num_epochs (int): Nombre d'époques d'entraînement.
        - args (namespace): Arguments avec des informations comme 'device' et autres paramètres.
    """
    log_dir = f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs saved in {log_dir}")

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, num_epochs, args, writer)
        eval(model, val_loader, criterion, args, writer, epoch)

    writer.close()
