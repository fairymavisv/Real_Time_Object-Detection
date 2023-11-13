from transformers import DetrFeatureExtractor, DetrForObjectDetection,DetrImageProcessor
from PIL import Image
import torch
import os
import json
import math
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrConfig
import pandas as pd

check_point = 'train_outmodel'

feature_extractor = DetrImageProcessor.from_pretrained(check_point)
# config = DetrConfig.from_pretrained(check_point, num_labels=2)
model = DetrForObjectDetection.from_pretrained(check_point)


def group_annotations_by_image_id(annotations):
    grouped_annotations = defaultdict(list)
    for annotation in annotations:
        grouped_annotations[annotation["image_id"]].append(annotation)
    return grouped_annotations

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# 将模型移动到GPU
model.to(device)
# eval

model.eval()

# 测试和评估模型
valid_annotations = json.load(open("./archive/valid_annotations"))
grouped_valid_annotations = group_annotations_by_image_id(valid_annotations)



def calculate_iou(box1, box2):
    """Calculate intersection over union value"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (box1_area + box2_area - intersection)


def calculate_center_distance(box1, box2):
    """Calculate distance between the center of two boxes"""
    box1_center = [(box1[2] + box1[0]) / 2, (box1[3] + box1[1]) / 2]
    box2_center = [(box2[2] + box2[0]) / 2, (box2[3] + box2[1]) / 2]

    return np.sqrt((box1_center[0] - box2_center[0]) ** 2 + (box1_center[1] - box2_center[1]) ** 2)



min_dist_images = []
successful_class_images = []

# 评估和可视化的代码
distances = []
ious = []
predicted_labels = []
true_labels = []
max_labels = []
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
for i in range(72):  # validate on 72 images
    img_path = f"./archive/valid/valid/image_id_{str(i).rjust(3, '0')}.jpg"
    image = Image.open(img_path)

    
    image_copy = image.copy()
    

    annotations_for_image = grouped_valid_annotations[i]
    coco_format_annotations = {"image_id": i, "annotations": annotations_for_image}


    inputs = feature_extractor(images=image, return_tensors="pt")
    # 将输入移动到GPU
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else [i.to(device) for i in v] for k, v in inputs.items()}


    outputs = model(**inputs)

    # 获得原始图片的尺寸
    target_sizes = torch.tensor([image.size[::-1]])  # 反转宽和高，因为PIL的size是宽和高的元组，但模型需要的是高和宽

    # postprocess predictions
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.01)[0]
    results = {k: v.cpu() for k, v in results.items()}
    max_score = 0
    max_label = None
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations_for_image:
        true_labels.append(annotation["category_id"])

        # 为真实边界框创建一个矩形补丁
        rect_true = patches.Rectangle((annotation["bbox"][0], annotation["bbox"][1]), annotation["bbox"][2],
                                      annotation["bbox"][3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_true)
        m = 0
        for idx,(score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            if score > max_score:
                max_score = score
                max_label = label
                m = idx
        if len(results["boxes"]) != 0:
            box = [round(i, 2) for i in results["boxes"][m].tolist()]
            # calculate metrics
            ious.append(calculate_iou(annotation["bbox"], box))
            distances.append(calculate_center_distance(annotation["bbox"], box))
            if results['labels'][m].item() == 16:
                predict_label =  'penguin'
            else:
                predict_label = 'turtle'

            
            # 存储图像和相关信息，以便之后进行可视化
            if len(min_dist_images) < 4:
                min_dist_images.append((image, max_score, predict_label, box, distances[-1]))
            else:
                # 查找最大距离的图像并替换
                max_dist_img_idx = max(range(4), key=lambda index: min_dist_images[index][4])
                if distances[-1] < min_dist_images[max_dist_img_idx][4]:
                    min_dist_images[max_dist_img_idx] = (image, max_score, predict_label, box, distances[-1])
            
            # print(
            #     f"label: {model.config.id2label[results['labels'][m].item()]} ,confidence: "
            #     f"{round(max_score.item(), 3)} , location: {box}"
            # )
            print(
                f"label: {predict_label} ,confidence: "
                f"{round(max_score.item(), 3)} , location: {box}"
            )

            rect_pred = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                          edgecolor='b', facecolor='none')
            ax.add_patch(rect_pred)
            # print(type(max_label))
            # max_label = max_label.numpy().tolist()[0]
            max_labels.append(max_label.item())
            if max_label == 16:
                max_label = 1
            else:
                max_label = 2
            # max
            predicted_labels.append(max_label)

        
        if max_label == annotation["category_id"]:
            if len(successful_class_images) < 16:
                successful_class_images.append((image_copy, predict_label))
        
    output_file = os.path.join(output_dir, f'image_{i}.png')
    plt.savefig(output_file)
    plt.close()

print('Average IOU:', np.mean(ious))
print('Standard Deviation of IOU:', np.std(ious))
print('Average Distance:', np.mean(distances))
print('Standard Deviation of Distance:', np.std(distances))

print(max_labels)
print(true_labels)
print(predicted_labels)
# 计算混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)

# 计算评估指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision_penguin = precision_score(true_labels, predicted_labels, pos_label=1)
recall_penguin = recall_score(true_labels, predicted_labels, pos_label=1)
f1_penguin = f1_score(true_labels, predicted_labels, pos_label=1)

precision_turtle = precision_score(true_labels, predicted_labels, pos_label=2)
recall_turtle = recall_score(true_labels, predicted_labels, pos_label=2)
f1_turtle = f1_score(true_labels, predicted_labels, pos_label=2)

print('Accuracy:', accuracy)
print('Precision for penguins:', precision_penguin)
print('Recall for penguins:', recall_penguin)
print('F1-score for penguins:', f1_penguin)
print('Precision for turtles:', precision_turtle)
print('Recall for turtles:', recall_turtle)
print('F1-score for turtles:', f1_turtle)

# 输出混淆矩阵
print("Confusion Matrix:")
print(cm)


df_cm = pd.DataFrame(cm,
  index = ['Actual Penguin', 'Actual Turtle'],
  columns = ['Predicted Penguin', 'Predicted Turtle']
)

# 使用seaborn绘制混淆矩阵热图
import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.show()


# 检测输出的可视化
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Detection Outputs")
for ax, (image, score, label, box, dist) in zip(axs.flatten(), min_dist_images):
    ax.imshow(image)
    rect_pred = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                  edgecolor='b', facecolor='none')
    ax.add_patch(rect_pred)
    ax.set_title(f"label: {label}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'detection_outputs.png'))

# 分类输出的可视化
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle("Classification Outputs")
for ax, (image, label) in zip(axs.flatten(), successful_class_images):
    ax.imshow(image)
    ax.set_title(f"Predicted Label: {label}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classification_outputs.png'))
