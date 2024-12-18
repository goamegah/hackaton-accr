import random

def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    return X_train, X_test, y_train, y_test