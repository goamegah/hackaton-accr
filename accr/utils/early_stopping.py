import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01, path='./checkpoint.pth'):
        """
        Args:
            patience (int): Nombre d'époques sans amélioration avant d'arrêter.
            min_delta (float): Minimum d'amélioration pour considérer une amélioration.
            path (str): Chemin où sauvegarder le meilleur modèle.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None
        self.path = path

    def __call__(self, val_loss, model):
        """
        Vérifie si l'entraînement doit être arrêté plus tôt en fonction de la performance du modèle.

        Args:
            val_loss (float): La perte de validation à l'époque courante.
            model (torch.nn.Module): Le modèle que l'on souhaite sauvegarder si sa performance est la meilleure.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Sauvegarde le modèle si la performance est la meilleure"""
        self.best_model_wts = model.state_dict()
        torch.save(self.best_model_wts, self.path)
