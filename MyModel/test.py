import time

# from train import *
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms

from MyModel.Models.AResUNet import AResUNet
from MyModel.Models.ResUNet import ResUNet
from MyModel.Models.Resnet_Deeplab import Resnet_Deeplab
from MyModel.dataset import Segmentation


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
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
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
    weights_path = '/root/autodl-tmp/save_weights/AResUNet_channel_5c_0.6236.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = AResUNet(num_classes=num_classes)
    # model = ResUNet(num_classes=num_classes)
    # model = DeepLabV3plus(num_classes=1, backbone="mobilenet", downsample_factor=16, pretrained=False)
    # model = Resnet_Deeplab(num_classes=1)
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()
    evaluator = Evaluator(num_class=2)
    evaluator.reset()
    # root_path = 'D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/'
    root_path = '../Test'

    image_folder = os.path.join(root_path, 'Images')
    mask_folder = os.path.join(root_path, 'Masks')

    images = os.listdir(image_folder)

    train_images, test_images = train_test_split(images, test_size=0.8, random_state=42)

    test_set = Segmentation(images, image_folder, mask_folder, train_val='test')

    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # sum_1 = 0  # 累加每张图片val的accuracy
    # sum_2 = 0  # 累积每张图片Val的mIoU
    #
    # image = Image.open("D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/Images/8.png")
    # label = cv2.imread("D:/OneDrive - The University of Nottingham/Dissertation/Data/ResearchData/Masks/8.png", cv2.IMREAD_GRAYSCALE)
    #
    # # orininal_h = image.shape[0]  # 读取的图像的高
    # # orininal_w = image.shape[1]  # 读取的图像的宽
    #
    # # image = cv2.resize(image, dsize=(256, 256))
    # label = cv2.resize(label, dsize=(256, 256))
    #
    # label[label >= 0.5] = 1  # label被resize后像素值会改变,调整像素值为原来的两类
    # label[label < 0.5] = 0
    #
    # image = img_transforms(image)
    # image = torch.unsqueeze(image, dim=0)
    #
    # predict = model(image.to(device)).cpu()
    #
    # q = np.squeeze(predict.data.numpy())
    # predict = torch.squeeze(predict, dim=0)  # [1,1,416,416]---->[1,416,416]
    # predict = predict.permute(1, 2, 0)
    # predict = predict.detach().numpy()
    # q[q < 0.5] = 0
    # q[q >= 0.5] = 1
    # prc = predict.argmax(axis=-1)
    #
    # save_full = os.path.join('./', 'haha.png')
    # cv2.imwrite(save_full, q * 255)
    # # 进行mIoU和accuracy的评测
    # imgPredict = q
    # imgLabel = label
    #
    # metric = Evaluator(2)
    # metric.add_batch(imgLabel,imgPredict)
    # acc = metric.Pixel_Accuracy_Class()
    # sum_1 += acc
    # mIoU = metric.Intersection_over_Union()
    # sum_2 += mIoU
    # # print("%s.jpg :" % test_images[i])
    # print("accuracy:  " + str(acc * 100) + " %")
    # print("mIoU:  " + str(mIoU))
    # print("-------------------")
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
    sum_f1 = 0
    sum_recall = 0
    sum_prec = 0
    sum_iou = 0
    s = 0
    with torch.no_grad():
        test_loader = DataLoader(
            test_set,
            batch_size=11,
            num_workers=16,
            shuffle=False,
            pin_memory=True
        )
        for images, masks in tqdm(test_loader):
            # img_height, img_width = images.shape[-2:]
            # init_img = torch.zeros((1, 2, img_height, img_width), device=device)
            # model(init_img)

            # t_start = time.time()
            # output = model(images.to(device)).cpu()
            output,weight = model(images.to(device))
            # t_end = time.time()
            # print("inference time: {}".format(t_end - t_start))
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
                # if i == 0 or i==1:
                #     continue
                # prediction = output[i].cpu()
                # predict = prediction.permute(1, 2, 0)
                # prc = predict.numpy()
                # prc = predict.argmax(axis=-1)
                prediction = output[i].cpu()
                prediction = np.squeeze(prediction.data.numpy())
                prc = prediction
                mask = masks[i].cpu()
                mask = torch.squeeze(mask)
                mask = mask.numpy().astype('uint8')
                # prediction = prediction.numpy()
                # prc = np.concatenate(prediction, axis=1)
                # prc = predictions[i].cpu().numpy()
                prc[prc < 0.5] = 0
                prc[prc >= 0.5] = 1
                evaluator.add_batch(pre_image=prc, gt_image=mask)
                # mask_name = masks[i]
                # results.append((prediction, str(mask_name)))
                acc = evaluator.Pixel_Accuracy_Class()
                sum_accuracy += acc
                f1 = evaluator.F1()
                sum_f1 += f1
                recall = evaluator.Recall()
                sum_recall += recall
                prec = evaluator.Precision()
                sum_prec += prec
                iou = evaluator.Intersection_over_Union()
                sum_iou += iou
                mIoU = evaluator.meanIntersectionOverUnion()
                sum_mIou += mIoU
                s += 1
                # print("%s:" % images.g)
                print("accuracy:  " + str(acc * 100) + " %")
                print("f1:  " + str(f1))
                print("recall:  " + str(recall))
                print("prec:  " + str(prec))
                print("iou:  " + str(iou))
                print("mIoU:  " + str(mIoU))
                print("-------------------")

        sum_accuracy = sum_accuracy / s
        sum_f1 = sum_f1 / s
        sum_recall = sum_recall / s
        sum_prec = sum_prec / s
        sum_iou = sum_iou / s
        sum_mIou = sum_mIou / s

        # sum_accuracy = round(sum_accuracy, 5)
        # sum_mIou = round(sum_mIou, 4)

        print("Mean accuracy:  " + str(sum_accuracy * 100) + " %")
        print("Mean F1:  " + str(sum_f1))
        print("Mean Recall:  " + str(sum_recall))
        print("Mean Precission:  " + str(sum_prec))
        print("Mean IoU:  " + str(sum_iou))
        print("Mean mIoU:  " + str(sum_mIou))


if __name__ == "__main__":
    main()
