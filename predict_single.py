
from segmentation import SegFormer_Segmentation

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

def analyze_segmentation_image(seg_image):
    """
    分析分割图像的颜色分布，帮助理解颜色映射
    """
    seg_array = np.array(seg_image)
    print(f"图像形状: {seg_array.shape}")
    print(f"数据类型: {seg_array.dtype}")
    print(f"数值范围: {seg_array.min()} - {seg_array.max()}")

    if len(seg_array.shape) == 3:
        # RGB图像
        print("这是一个RGB图像")
        # 统计不同的颜色
        colors = seg_array.reshape(-1, 3)
        unique_colors = np.unique(colors, axis=0)
        print(f"唯一颜色数量: {len(unique_colors)}")
        print("前10个唯一颜色:")
        for i, color in enumerate(unique_colors[:10]):
            count = np.sum(np.all(colors == color, axis=1))
            print(f"  颜色 {i}: RGB{tuple(color)} - 像素数: {count}")
    else:
        # 灰度图像
        print("这是一个灰度图像")
        unique_values = np.unique(seg_array)
        print(f"唯一值: {unique_values}")
        for val in unique_values:
            count = np.sum(seg_array == val)
            print(f"  值 {val}: 像素数: {count}")

    return seg_array

def rgb_to_class_prediction(seg_array, color_mapping):
    """
    将RGB图像转换为类别预测 - 优化版本
    """
    if len(seg_array.shape) == 3:
        seg_pred = np.zeros((seg_array.shape[0], seg_array.shape[1]), dtype=np.uint8)

        # 获取所有唯一颜色及其位置
        colors = seg_array.reshape(-1, 3)
        unique_colors, inverse_indices = np.unique(colors, axis=0, return_inverse=True)

        # 为每种唯一颜色分配类别
        class_assignments = np.zeros(len(unique_colors), dtype=np.uint8)
        for i, color in enumerate(unique_colors):
            color_tuple = tuple(color)
            if color_tuple in color_mapping:
                class_assignments[i] = color_mapping[color_tuple]
            else:
                print(f"警告: 颜色 RGB{color_tuple} 不在映射中")

        # 重建预测数组
        seg_pred_flat = class_assignments[inverse_indices]
        seg_pred = seg_pred_flat.reshape(seg_array.shape[:2])

        return seg_pred
    else:
        # 灰度图像，直接返回
        return seg_array

def add_text_labels_debug(ax, seg_pred, name_list, min_area_threshold=200):
    """
    带调试信息的文本标签添加函数
    """
    unique_classes = np.unique(seg_pred)
    print(f"分割结果中的唯一类别: {unique_classes}")

    for class_id in unique_classes:
        if class_id >= len(name_list):
            print(f"警告: 类别ID {class_id} 超出名称列表范围")
            continue

        class_mask = (seg_pred == class_id)
        area = np.sum(class_mask)
        print(f"类别 {class_id} ({name_list[class_id]}): 面积 = {area}")

        if area < min_area_threshold:
            print(f"  跳过 - 面积小于阈值 {min_area_threshold}")
            continue

        # 使用连通组件分析
        labeled_array, num_features = ndimage.label(class_mask)
        if num_features > 0:
            component_sizes = ndimage.sum(class_mask, labeled_array, range(1, num_features + 1))
            largest_component = np.argmax(component_sizes) + 1
            largest_mask = (labeled_array == largest_component)

            center_of_mass = ndimage.center_of_mass(largest_mask)
            if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
                y, x = center_of_mass
                class_name = name_list[class_id]
                if ',' in class_name:
                    class_name = class_name.split(',')[0]

                print(f"  添加标签 '{class_name}' 在位置 ({x:.1f}, {y:.1f})")

                ax.text(x, y, class_name,
                        color='white',
                        fontsize=20,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='black',
                                  alpha=0.7,
                                  edgecolor='white',
                                  linewidth=1))

# 使用示例 - 针对你的具体配置优化
def display_segmentation_with_labels_optimized(seg_image, name_classes, min_area_threshold=200):
    """
    专门为你的颜色配置优化的显示函数
    """
    # 你的具体颜色映射
    color_mapping = {
        (255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5
    }

    # default_color_mapping = {
    #     (100, 149, 237): 0,  # background - 蓝色
    #     (102, 205, 170): 1,  # building - 青色
    #     (205, 133, 63): 2,  # road - 棕色
    #     (160, 32, 240): 3,  # water - 紫色
    #     (0, 200, 0): 4,  # barren - 绿色
    #     (139, 69, 19): 5,  # forest - 深棕色
    #     (0, 255, 0): 6  # agriculture - 亮绿色
    # }

    seg_array = np.array(seg_image)
    print(f"图像形状: {seg_array.shape}")

    # 快速检查是否有精确匹配
    colors = seg_array.reshape(-1, 3)
    unique_colors = np.unique(colors, axis=0)
    print(f"图像中的唯一颜色数量: {len(unique_colors)}")

    # 转换为类别预测
    seg_pred = np.zeros((seg_array.shape[0], seg_array.shape[1]), dtype=np.uint8)

    for color in unique_colors:
        color_tuple = tuple(color)
        mask = np.all(seg_array == color, axis=2)
        pixel_count = np.sum(mask)

        if color_tuple in color_mapping:
            class_id = color_mapping[color_tuple]
            seg_pred[mask] = class_id
            class_name = name_classes[class_id] if class_id < len(name_classes) else "未知"
            print(f"精确匹配: RGB{color_tuple} -> 类别 {class_id} ({class_name}), 像素数: {pixel_count}")
        else:
            # 寻找最接近的颜色
            min_distance = float('inf')
            best_class = 0
            for mapped_color, class_id in color_mapping.items():
                distance = np.sqrt(np.sum((np.array(color) - np.array(mapped_color)) ** 2))
                if distance < min_distance:
                    min_distance = distance
                    best_class = class_id

            seg_pred[mask] = best_class
            class_name = name_classes[best_class] if best_class < len(name_classes) else "未知"
            print(
                f"近似匹配: RGB{color_tuple} -> 类别 {best_class} ({class_name}), 距离: {min_distance:.2f}, 像素数: {pixel_count}")

    # 显示结果
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(seg_image)
    ax.axis('off')
    # 添加文本标签
    add_text_labels_debug(ax, seg_pred, name_classes, min_area_threshold)

    plt.tight_layout()
    plt.show()

    return seg_pred


# 使用示例
if __name__ == "__main__":
    segformer = SegFormer_Segmentation()
    mode = "predict"
    count = False
    name_classes = ["Impervious surfaces", "Building", "Low vegetation", "Tree", "Car", "background"]
    # name_classes = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
    simplify = True

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = segformer.detect_image(image, count=count, name_classes=name_classes)

            # 使用优化版本（推荐）
            seg_pred = display_segmentation_with_labels_optimized(r_image, name_classes, min_area_threshold=200)
