import time

import cv2
import numpy as np

from train import *
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        mIoU = round(mIoU, 4)
        return mIoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        acc = round(acc, 5)
        return acc

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def main():
    num_classes = 1
    weights_path = './pre_weights/resUNet_checkpoint_0.1891.pth'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # model = ResUNet(num_classes=num_classes)
    model = DeepLabV3plus(num_classes=1, backbone="mobilenet", downsample_factor=16, pretrained=False)
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()
    evaluator = Evaluator(num_class=2)
    evaluator.reset()
    root_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/'

    image_folder = os.path.join(root_path, 'Images')
    mask_folder = os.path.join(root_path, 'Masks')

    images = os.listdir(image_folder)

    train_images, test_images = train_test_split(images, test_size=0.8, random_state=42)

    test_set = Segmentation(test_images, image_folder, mask_folder, train_val='test')

    # sum_1 = 0  # 累加每张图片val的accuracy
    # sum_2 = 0  # 累积每张图片Val的mIoU
    #
    #
    # for i in range(len(test_images)):
    #     image = cv2.imread("D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/Images/%s" % test_images[i], -1)
    #     label = cv2.imread("D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/Masks/%s" % test_images[i], -1)
    #
    #     orininal_h = image.shape[0]  # 读取的图像的高
    #     orininal_w = image.shape[1]  # 读取的图像的宽
    #
    #     image = cv2.resize(image, dsize=(256, 256))
    #     label = cv2.resize(label, dsize=(256, 256))
    #
    #     label[label >= 0.5] = 1  # label被resize后像素值会改变,调整像素值为原来的两类
    #     label[label < 0.5] = 0
    #
    #     image = image / 255.0  # 图像归一化
    #     image = torch.from_numpy(image)
    #     image = image.permute(2, 0, 1)  # 显式的调转维度
    #
    #     image = torch.unsqueeze(image, dim=0)  # 改变维度,使得符合model input size
    #
    #     predict = model(image.to(device, dtype=torch.float32)).cpu()
    #
    #
    #     predict = torch.squeeze(predict,dim=0)  # [1,1,416,416]---->[1,416,416]
    #     predict = predict.permute(1, 2, 0)
    #
    #     predict = predict.detach().numpy()
    #
    #     prc = predict.argmax(axis=-1)
    #
    #     # 进行mIoU和accuracy的评测
    #     imgPredict = prc
    #     imgLabel = label
    #
    #     metric = Evaluator(2)
    #     metric.add_batch(imgPredict, imgLabel)
    #     acc = metric.Pixel_Accuracy_Class()
    #     sum_1 += acc
    #     mIoU = metric.meanIntersectionOverUnion()
    #     sum_2 += mIoU
    #     print("%s.jpg :" % test_images[i])
    #     print("accuracy:  " + str(acc * 100) + " %")
    #     print("mIoU:  " + str(mIoU))
    #     print("-------------------")
    #
    # # 全部图片平均的accuracy和mIoU
    # sum_1 = sum_1 / len(test_images)
    # sum_2 = sum_2 / len(test_images)
    #
    # sum_1 = round(sum_1, 5)
    # sum_2 = round(sum_2, 4)
    #
    # print("M accuracy:  " + str(sum_1 * 100) + " %")
    # print("M mIoU:  " + str(sum_2))

    sum_accuracy = 0
    sum_mIou = 0
    with torch.no_grad():
        test_loader = DataLoader(
            test_set,
            batch_size=2,
            num_workers=0,
            shuffle=True,
            pin_memory=True
        )
        for images, masks in tqdm(test_loader):
            img_height, img_width = images.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time.time()
            output = model(images.to(device, dtype=torch.float32)).cpu()
            t_end = time.time()
            print("inference time: {}".format(t_end - t_start))
            masks[masks >= 0.5] = 1
            masks[masks < 0.5] = 0

            # output = output.permute(0, 2, 3, 1)
            # output = output.numpy()
            # prediction = np.concatenate(output, axis=1)
            # predictions = output.argmax(dim=1)
            # predictions = torch.squeeze(output)
            # masks = torch.squeeze(masks)
            results = []
            for i in range(output.shape[0]):

                # prediction = output[i].cpu()
                # predict = prediction.permute(1, 2, 0)
                # prc = predict.numpy()
                # prc = predict.argmax(axis=-1)
                prediction = output[i].cpu()
                prediction = prediction.permute(1, 2, 0)
                prediction = prediction.numpy()
                prc = np.concatenate(prediction, axis=1)
                # prc = predictions[i].cpu().numpy()
                prc = np.where(prc >= 0.5, 1, 0)
                evaluator.add_batch(pre_image=prc, gt_image=masks[i][0].cpu().numpy())
                # mask_name = masks[i]
                # results.append((prediction, str(mask_name)))
                acc = evaluator.Pixel_Accuracy_Class()
                sum_accuracy += acc
                mIoU = evaluator.meanIntersectionOverUnion()
                sum_mIou += mIoU
                # print("%s:" % images.g)
                print("accuracy:  " + str(acc * 100) + " %")
                print("mIoU:  " + str(mIoU))
                print("-------------------")

        sum_accuracy = sum_accuracy / len(test_loader)
        sum_mIou = sum_mIou / len(test_loader)

        sum_accuracy = round(sum_accuracy, 5)
        sum_mIou = round(sum_mIou, 4)

        print("Mean accuracy:  " + str(sum_accuracy * 100) + " %")
        print("Mean IoU:  " + str(sum_mIou))


if __name__ == "__main__":
    main()
