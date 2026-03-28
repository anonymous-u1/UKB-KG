import numpy as np

class ModelSaving:
    """
    Simple early stopping utility.
    Stops training if validation loss does not improve after `patience` epochs.
    """
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            # Validation set loss decreases, reset counter.
            self.best_loss = val_loss
            self.counter = 0
        else:
            # loss not improved
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
