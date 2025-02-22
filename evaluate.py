import os

from PIL import Image
from tqdm import tqdm

from segmentation import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，所以看到近似全黑的图是正常的。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   类别个数
    #------------------------------#
    num_classes     = 5
    #--------------------------------------------#
    name_classes = ["Impervious surfaces", "Building", "Low vegetation", "Tree", "Car"]
    # name_classes = ["background", "building", "road", "barren", "water", "forest", "agriculture"]  # loveda数据集
    #-------------------------------------------------------#
    #   数据集文件夹
    #-------------------------------------------------------#
    Dataset_path  = 'YourDataset'

    image_ids       = open(os.path.join(Dataset_path, "ImageSets/Segmentation/test.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(Dataset_path, "SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        segformer = SegFormer_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(Dataset_path, "JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = segformer.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
