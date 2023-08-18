import torch
import os
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MyModel.Models.AResUNet import AResUNet
from MyModel.Models.ResUNet import ResUNet
from MyModel.test import Evaluator
from dataset import Segmentation
from sklearn.model_selection import train_test_split

from MyModel.Loss.DICE_BCE_Loss import dice_coeff, DICE_BCE_Loss, BinaryDiceLoss
from MyModel.Models.DeepLabV3plus import DeepLabV3plus
from MyModel.Models.Resnet_Deeplab import Resnet_Deeplab
from MyModel.Models.UNet import UNet


import matplotlib.pyplot as plt
import numpy as np

from deepflash2 import losses as L


def createModel(num_classes, pretrain=False):
    # model = UNet(5, num_classes)
    # model = DeepLabV3plus(num_classes=num_classes, backbone="mobilenet", downsample_factor=16, pretrained=pretrain)
    # model = ResUNet(num_classes=num_classes)
    model = AResUNet(num_classes=num_classes)
    # model = Resnet_Deeplab(num_classes=1)
    if pretrain:
        weights_dict = torch.load('/root/autodl-tmp/save_weights/AResUnet_5c_checkpoint_0.6281.pth', map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)
    return model


def train(epochs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    ################### Dataset & DataLoader ##############################
    # root_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/Water_Bodies_Dataset/'
    # root_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/'
    root_path = '../ResearchData'
    # root_path = '../DataGroup'
    # root_path = '../ResearchData_5C'
    # root_path = 'D:\\OneDrive - The University of Nottingham\\Dissertation\\Data\\DataGroup\\'

    image_folder = os.path.join(root_path, 'Images')
    mask_folder = os.path.join(root_path, 'Masks')

    images = os.listdir(image_folder)

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    # test_images, val_images = train_test_split(test_images, test_size=0.5, random_state=42)

    train_set = Segmentation(train_images, image_folder, mask_folder, train_val='train')
    # test_set = Segmentation(test_images, image_folder, mask_folder, train_val='test')
    val_set = Segmentation(val_images, image_folder, mask_folder, train_val='val')

    train_loader = DataLoader(train_set, batch_size=11, num_workers=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=11, num_workers=16, shuffle=True, pin_memory=True)
    # test_loader = DataLoader(test_set, batch_size=8, num_workers=16, shuffle=True, pin_memory=True)
    # train_loader = DataLoader(train_set, batch_size=2, num_workers=0)
    # val_loader = DataLoader(val_set, batch_size=2, num_workers=0)
    # test_loader = DataLoader(test_set, batch_size=1, num_workers=0)

    ############################## Model & optimizer & scheduler ##############################

    model = createModel(1, pretrain=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,
        T_mult=2,
        eta_min=1e-6
    )
    # loss = DICE_BCE_Loss()
    # loss = nn.functional.cross_entropy()
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = BinaryDiceLoss()
    loss_fn = L.JointLoss(first=dice_loss, second=bce_loss, first_weight=0.5, second_weight=0.5).cuda()

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    writer = SummaryWriter("logs")
    total_train_step = 0
    total_test_step = 0
    outWeight = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits, weight = model(images)
            outWeight = weight
            logits = torch.squeeze(logits)
            masks = torch.squeeze(masks)
            l = loss_fn(logits, masks)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_dice += dice_coeff(logits, masks)
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("number of training: {}, Loss: {}".format(total_train_step, l.item()))
        # writer.add_scalar("train_loss", train_show, epoch)
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)
                logits, v_weight = model(images)
                logits = torch.squeeze(logits)
                masks = torch.squeeze(masks)
                l = loss_fn(logits, masks)
                val_loss += l.item()
                val_dice += dice_coeff(logits, masks)
                total_test_step += 1
                if total_test_step % 100 == 0:
                    print("number of val: {}, Loss: {}".format(total_test_step, l.item()))

        # print("accuracy in test_data: {}".format(total_accuracy / test_data_size))

        total_test_step += 1
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        writer.add_scalars("loss",{"train_loss": train_loss,"val_loss": val_loss},epoch)
        writer.add_scalars("dice", {"train_dice": train_dice, "val_dice": val_dice}, epoch)
        # writer.add_scalar("loss/val_loss", val_loss, epoch)
        # writer.add_scalar("dice/train_dice", train_dice, epoch)
        # writer.add_scalar("dice/val_dice", val_dice, epoch)
        print(
            f"Epoch: {epoch + 1}  Train Loss: {train_loss:.4f} | Train DICE Coeff: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val DICE Coeff: {val_dice:.4f}")

        # if (epoch+1) % 10 == 0:
        #     save_file = {"model": model.state_dict(),
        #                  "optimizer": optimizer.state_dict(),
        #                  "lr_scheduler": scheduler.state_dict(),
        #                  "epoch": epoch + 1}
        #     torch.save(save_file, f"/root/autodl-tmp/save_weights/AResUNet_5c_checkpoint_{train_loss:.4f}.pth")

        save_interval = 10  # int(max_epoch/6)
        if epoch > int(epochs / 2) and (epoch + 1) % save_interval == 0:
            print(outWeight)
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": scheduler.state_dict(),
                         "epoch": epoch + 1}
            torch.save(save_file, f"/root/autodl-tmp/save_weights/AResUNet_channel_5c_{train_loss:.4f}.pth")

        if epoch >= epochs - 1:
            print(outWeight)
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": scheduler.state_dict(),
                         "epoch": epoch + 1}
            torch.save(save_file, f"/root/autodl-tmp/save_weights/AResUNet_channel_5c_{train_loss:.4f}.pth")
            break

        scheduler.step()

    plt.figure(figsize=(10,6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), train_dices)
    plt.plot(np.arange(epochs), val_dices)
    plt.xlabel("Epoch")
    plt.ylabel("DICE Coeff")
    plt.legend(["Train DICE", "Val DICE"])
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), train_losses)
    plt.plot(np.arange(epochs), val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train Loss", "Val Loss"])
    plt.show()


if __name__ == '__main__':
    epochs = 60
    train(epochs=epochs)