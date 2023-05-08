import torch
import torch.nn as nn
from torch.utils.data import Dataset
import lightning.pytorch as L
from torchmetrics.classification import MultilabelAccuracy


class ListDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

    def __len__(self):
        return len(self.list1)

    def __getitem__(self, idx):
        item1 = self.list1[idx]
        item2 = self.list2[idx]
        return item1, item2

class CNN_mel(L.LightningModule):
    def __init__(self, lr, num_labels, aggregation_method="s1"):
        super(CNN_mel, self).__init__()
        self.train_acc = MultilabelAccuracy(num_labels=num_labels)
        self.val_acc = MultilabelAccuracy(num_labels=num_labels)
        self.lr = lr
        self.aggregation_method = aggregation_method
        self.model_predictions = []

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.33)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.33)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.33)
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.33)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.33)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.33)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.leakyrelu7 = nn.LeakyReLU(negative_slope=0.33)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.leakyrelu8 = nn.LeakyReLU(negative_slope=0.33)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout4 = nn.Dropout2d(p=0.25)

        self.globalpool = nn.AdaptiveMaxPool2d((1,1))

        self.fc1 = nn.Linear(512, 1024)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.leakyrelu5(x)
        x = self.conv6(x)
        x = self.leakyrelu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.leakyrelu7(x)
        x = self.conv8(x)
        x = self.leakyrelu8(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = self.globalpool(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.dropout5(x)
        x = self.fc2(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.train_acc.update(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': self.log}
    
    def on_train_epoch_end(self):
        train_accuracy = self.train_acc.compute()
        self.log("train_accuracy", train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.val_acc.update(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'log': self.log}
    
    def on_validation_epoch_end(self):
        val_accuracy = self.val_acc.compute()
        self.log("val_accuracy", val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx):
        key, val = batch
        val = val.squeeze(dim=0)
        predictions = []
        for i in range(val.size(0)):
            x = val[i]
            y_hat = self.forward(x)
            soft = nn.Softmax(dim=1)
            y_hat = soft(y_hat)
            predictions.append(y_hat)
        predictions = torch.stack(predictions)
        predictions = predictions.squeeze(dim=1)

        if self.aggregation_method == "s1":
            predictions = torch.mean(predictions, axis = 0)
        elif self.aggregation_method == "s2":
            predictions = torch.sum(predictions, axis = 0)
            m = torch.max(predictions)
            predictions /= m
        
        self.model_predictions.append(predictions)


class CNN_modgd(L.LightningModule):
    def __init__(self, lr, num_labels, aggregation_method="s1"):
        super(CNN_modgd, self).__init__()
        self.train_acc = MultilabelAccuracy(num_labels=num_labels)
        self.val_acc = MultilabelAccuracy(num_labels=num_labels)
        self.lr = lr
        self.aggregation_method = aggregation_method
        self.model_predictions = []

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.33)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.33)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.33)
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.33)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.33)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.33)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.leakyrelu7 = nn.LeakyReLU(negative_slope=0.33)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.leakyrelu8 = nn.LeakyReLU(negative_slope=0.33)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout4 = nn.Dropout2d(p=0.25)

        self.globalpool = nn.AdaptiveMaxPool2d((1,1))

        self.fc1 = nn.Linear(512, 1024)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.leakyrelu5(x)
        x = self.conv6(x)
        x = self.leakyrelu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.leakyrelu7(x)
        x = self.conv8(x)
        x = self.leakyrelu8(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = self.globalpool(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.dropout5(x)
        x = self.fc2(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.train_acc.update(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': self.log}
    
    def on_train_epoch_end(self):
        train_accuracy = self.train_acc.compute()
        self.log("train_accuracy", train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.val_acc.update(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'log': self.log}
    
    def on_validation_epoch_end(self):
        val_accuracy = self.val_acc.compute()
        self.log("val_accuracy", val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx):
        key, val = batch
        val = val.squeeze(dim=0)
        predictions = []
        for i in range(val.size(0)):
            x = val[i]
            y_hat = self.forward(x)
            soft = nn.Softmax(dim=1)
            y_hat = soft(y_hat)
            predictions.append(y_hat)
        predictions = torch.stack(predictions)
        predictions = predictions.squeeze(dim=1)

        if self.aggregation_method == "s1":
            predictions = torch.mean(predictions, axis = 0)
        elif self.aggregation_method == "s2":
            predictions = torch.sum(predictions, axis = 0)
            m = torch.max(predictions)
            predictions /= m
        
        self.model_predictions.append(predictions)
    
class CNN_pitch(L.LightningModule):
    def __init__(self, lr, num_labels, aggregation_method="s1"):
        super(CNN_pitch, self).__init__()
        self.train_acc = MultilabelAccuracy(num_labels=num_labels)
        self.val_acc = MultilabelAccuracy(num_labels=num_labels)
        self.lr = lr
        self.aggregation_method = aggregation_method
        self.model_predictions = []
        
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 3), padding='same')
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.33)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=(3, 3), padding='same')
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.33)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.33)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.33)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same')
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.33)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same')
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.33)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout3 = nn.Dropout(p=0.25)
        
        self.gpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        
        self.fc1 = nn.Linear(512, 1024)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 11)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv5(x)
        x = self.leakyrelu5(x)
        x = self.conv6(x)
        x = self.leakyrelu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.gpool(x)
        
        x = x.view(-1, 512)
        
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.train_acc.update(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': self.log}
    
    def on_train_epoch_end(self):
        train_accuracy = self.train_acc.compute()
        self.log("train_accuracy", train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.val_acc.update(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'log': self.log}
    
    def on_validation_epoch_end(self):
        val_accuracy = self.val_acc.compute()
        self.log("val_accuracy", val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx):
        key, val = batch
        val = val.squeeze(dim=0)
        predictions = []
        for i in range(val.size(0)):
            x = val[i]
            y_hat = self.forward(x)
            soft = nn.Softmax(dim=1)
            prediction = soft(y_hat)
            predictions.append(prediction)
        predictions = torch.stack(predictions)
        predictions = predictions.squeeze(dim=1)

        if self.aggregation_method == "s1":
            predictions = torch.mean(predictions, axis = 0)
        elif self.aggregation_method == "s2":
            predictions = torch.sum(predictions, axis = 0)
            m = torch.max(predictions)
            predictions /= m
        
        self.model_predictions.append(predictions)