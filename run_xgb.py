import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb  # Importation de XGBoost

parser = argparse.ArgumentParser(description='XGBoost Classifier')

parser.add_argument('-j', '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='seed for initializing training.')

parser.add_argument('--early-stopping',
                    action='store_true',
                    help='Enable early stopping')

def main():
    args = parser.parse_args()

    # Fix the random seed for reproducibility
    np.random.seed(args.seed)

    # Charger les données prétraitées
    train_df = pd.read_csv('./dataset/preprocessed_data.csv')

    # Conversion de toutes les colonnes en int, tout en gérant les erreurs pour les colonnes non numériques
    train_ds = train_df.apply(pd.to_numeric, errors='coerce')

    # Séparation des features (X) et de la target (y)
    y = train_ds['grav'] - 1  # Ajuster les labels pour qu'ils commencent à 0
    X = train_ds.drop(columns=['grav'])

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    # Normalisation des données (optionnelle pour XGBoost, mais utile si les échelles des features varient beaucoup)
    X_train = (X_train - np.mean(X_train)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_train)) / np.std(X_train, axis=0)

    # Création des DMatrix pour XGBoost (XGBoost utilise ce format pour accélérer les calculs)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Paramètres du modèle XGBoost
    params = {
        'objective': 'multi:softmax',  # Prédiction de classes discrètes
        'num_class': len(np.unique(y)),  # Nombre de classes dans la target
        'eta': 0.1,  # Taux d'apprentissage
        'max_depth': 6,  # Profondeur maximale de chaque arbre
        'seed': args.seed,  # Graine de répétabilité
        'eval_metric': 'mlogloss'  # Fonction de perte (log-loss multinomiale)
    }

    # Dictionnaire des ensembles d'entraînement et de validation
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    # Early stopping éventuel
    early_stopping_rounds = 10 if args.early_stopping else None

    # Entraînement du modèle XGBoost
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,  # Nombre maximal de rounds d'optimisation
        evals=evals,  # Suivi de la performance sur le train et le test
        early_stopping_rounds=early_stopping_rounds,  # Arêt précoce si la perte ne s'améliore pas
        verbose_eval=True  # Affichage de l'évolution des performances
    )

    # Sauvegarder le modèle
    model.save_model('./xgboost_model.json')

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(dtest)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
