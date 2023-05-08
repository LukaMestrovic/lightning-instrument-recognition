from hydra import compose, initialize
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_modules import CNN_mel, CNN_modgd, CNN_pitch

# global initialization
initialize(version_base=None, config_path="../../configs")
cfg = compose(config_name="config")

#hyper parameters
epochs = cfg.training.epochs
batch_size = cfg.training.batch_size
lr = cfg.training.learning_rate
num_labels = cfg.constants.num_classes
num_workers = os.cpu_count()

#models
models = cfg.training.models

        
if __name__ == '__main__':

    if "mel" in models:

        X_train_mel = torch.load('../../data/processed/X_train_mel.pt')
        X_val_mel = torch.load('../../data/processed/X_val_mel.pt')
        y_train = torch.load('../../data/processed/y_train.pt')
        y_val = torch.load('../../data/processed/y_val.pt')
        train_dataset = TensorDataset(X_train_mel, y_train)
        val_dataset = TensorDataset(X_val_mel, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers , shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers , shuffle=False)
        
        model = CNN_mel(lr, num_labels)
        trainer = L.Trainer(callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath='../../models/', filename='cnn_mel')],
                            max_epochs=epochs, fast_dev_run=False)
        trainer.fit(model, train_loader, val_loader)

    if "modgd" in models:

        X_train_modgd = torch.load('../../data/processed/X_train_modgd.pt')
        X_val_modgd = torch.load('../../data/processed/X_val_modgd.pt')
        y_train = torch.load('../../data/processed/y_train.pt')
        y_val = torch.load('../../data/processed/y_val.pt')
        train_dataset = TensorDataset(X_train_modgd, y_train)
        val_dataset = TensorDataset(X_val_modgd, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers , shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers , shuffle=False)

        model = CNN_modgd(lr, num_labels)
        trainer = L.Trainer(callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath='../../models/', filename='cnn_modgd')],
                            max_epochs=epochs, fast_dev_run=False)
        trainer.fit(model, train_loader, val_loader)

    if "pitch" in models:

        X_train_pitch = torch.load('../../data/processed/X_train_pitch.pt')
        X_val_pitch = torch.load('../../data/processed/X_val_pitch.pt')
        y_train = torch.load('../../data/processed/y_train.pt')
        y_val = torch.load('../../data/processed/y_val.pt')
        train_dataset = TensorDataset(X_train_pitch, y_train)
        val_dataset = TensorDataset(X_val_pitch, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers , shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers , shuffle=False)

        model = CNN_pitch(lr, num_labels)
        trainer = L.Trainer(callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath='../../models/', filename='cnn_pitch')],
                            max_epochs=epochs, fast_dev_run=False)
        trainer.fit(model, train_loader, val_loader)