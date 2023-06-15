import torch
import os
from tqdm import tqdm
from Loss.DICE_BCE_Loss import dice_coeff, DICE_BCE_Loss
from Models.UNet import UNet
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import Segmentation
from sklearn.model_selection import train_test_split


def createModel(num_classes, pretrain=True):
    model = UNet(3, num_classes)
    if pretrain:
        weights_dict = torch.load('./pre_weights/water_bodies_model.pth', map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)
    return model


def train(epochs=10):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    ################### Dataset & DataLoader ##############################
    root_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/Water Bodies Dataset/'

    image_folder = os.path.join(root_path, 'Images')
    mask_folder = os.path.join(root_path, 'Masks')

    images = os.listdir(image_folder)

    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    test_images, val_images = train_test_split(test_images, test_size=0.5, random_state=42)

    train_set = Segmentation(train_images, image_folder, mask_folder, train_val='train')
    test_set = Segmentation(test_images, image_folder, mask_folder, train_val='test')
    val_set = Segmentation(val_images, image_folder, mask_folder, train_val='val')

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

    ############################## Model & optimizer & scheduler ##############################

    model = createModel(1, pretrain=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = DICE_BCE_Loss()
    # loss = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    writer = SummaryWriter("logs")
    total_train_step = 0
    total_test_step = 0
    train_data_size = len(train_loader)
    test_data_size = len(val_loader)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, masks)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_dice += dice_coeff(logits, masks)
            total_train_step += 1
            if total_train_step % 10 == 0:
                writer.add_scalar("train_loss", l.item(), total_train_step)
                print("number of training: {}, Loss: {}".format(total_train_step, l.item()))
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
                logits = model(images)
                l = loss(logits, masks)
                val_loss += l.item()
                val_dice += dice_coeff(logits, masks)
                accuracy = (logits.argmax(1) == masks).sum()
                total_accuracy += accuracy

        print("accuracy in test_data: {}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)

        total_test_step += 1
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        print(
            f"Epoch: {epoch + 1}  Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f}")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     # "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch}
        torch.save(save_file, f"save_weights/model_checkpoint_{train_loss:.4f}.pth")

if __name__ == '__main__':
    epochs = 30
    train(epochs=epochs)
