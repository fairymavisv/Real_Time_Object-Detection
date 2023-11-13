from transformers import DetrFeatureExtractor, DetrForObjectDetection,DetrImageProcessor
from PIL import Image
import torch
import os
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrConfig
import pandas as pd



check_point = 'model'
feature_extractor = DetrImageProcessor.from_pretrained(check_point)
# config = DetrConfig.from_pretrained(check_point, num_labels=2)
model = DetrForObjectDetection.from_pretrained(check_point)
# model.class_labels_classifier = torch.nn.Linear(in_features=model.class_labels_classifier.in_features, out_features=2)

# model.config.id2label = {0: "Class 1", 1: "Class 2"}
# model.config.label2id = {"Class 1": 0, "Class 2": 1}
# model.class_labels_classifier = torch.nn.Linear(256, 2)  # 修改最后一层
# id2label = {0: "Class 1", 1: "Class 2"}
# label2id = {"Class 1": 0, "Class 2": 1}
#
# # 将映射应用到模型配置
# model.config.id2label = id2label
# model.config.label2id = label2id






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

# 训练模型
model.train()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 50

def group_annotations_by_image_id(annotations):
    grouped_annotations = defaultdict(list)
    for annotation in annotations:
        grouped_annotations[annotation["image_id"]].append(annotation)
    return grouped_annotations

train_annotations = json.load(open("./archive/train_annotations"))
grouped_train_annotations = group_annotations_by_image_id(train_annotations)
loss_values = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(500):  # train on 500 images
        img_path = f"./archive/train/train/image_id_{str(i).rjust(3,'0')}.jpg"
        image = Image.open(img_path)

        annotations_for_image = grouped_train_annotations[i]
        if annotations_for_image[0]['category_id'] == 1 :
            annotations_for_image[0]['category_id'] = 16
        else:
            annotations_for_image[0]['category_id'] = 0
        coco_format_annotations = {"image_id": i, "annotations": annotations_for_image}

        inputs = feature_extractor(images=image, annotations=coco_format_annotations, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else [i.to(device) for i in v] for k, v in inputs.items()}

        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / 500
    loss_values.append(average_epoch_loss)
    print(f"Epoch {epoch + 1}, Loss: {average_epoch_loss}")

# plot the loss values
plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.plot(loss_values)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()








# # eval
# model.save_pretrained("train_outmodel")
# model.eval()
#

# valid_annotations = json.load(open("./archive/valid_annotations"))
# grouped_valid_annotations = group_annotations_by_image_id(valid_annotations)
#

# def calculate_iou(box1, box2):
#     """Calculate intersection over union value"""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#
#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#
#     return intersection / (box1_area + box2_area - intersection)
#
#
# def calculate_center_distance(box1, box2):
#     """Calculate distance between the center of two boxes"""
#     box1_center = [(box1[2] + box1[0]) / 2, (box1[3] + box1[1]) / 2]
#     box2_center = [(box2[2] + box2[0]) / 2, (box2[3] + box2[1]) / 2]
#
#     return np.sqrt((box1_center[0] - box2_center[0]) ** 2 + (box1_center[1] - box2_center[1]) ** 2)
#

# distances = []
# ious = []
# predicted_labels = []
# true_labels = []
# output_dir = 'output'
# os.makedirs(output_dir, exist_ok=True)
# for i in range(72):  # validate on 72 images
#     img_path = f"./archive/valid/valid/image_id_{str(i).rjust(3, '0')}.jpg"
#     image = Image.open(img_path)
#
#     annotations_for_image = grouped_valid_annotations[i]
#     coco_format_annotations = {"image_id": i, "annotations": annotations_for_image}
#
#
#     inputs = feature_extractor(images=image, return_tensors="pt")

#     inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else [i.to(device) for i in v] for k, v in inputs.items()}
#
#
#     outputs = model(**inputs)

#     target_sizes = torch.tensor([image.size[::-1]])  # 反转宽和高，因为PIL的size是宽和高的元组，但模型需要的是高和宽
#
#     # postprocess predictions
#     results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
#     results = {k: v.cpu() for k, v in results.items()}
#     max_score = 0
#     max_label = None
#     fig, ax = plt.subplots(1)
#     ax.imshow(image)
#
#     for annotation in annotations_for_image:
#         true_labels.append(annotation["category_id"])
#
#         rect_true = patches.Rectangle((annotation["bbox"][0], annotation["bbox"][1]), annotation["bbox"][2],
#                                       annotation["bbox"][3], linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect_true)
#         m = 0
#         for idx,(score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
#             if score > max_score:
#                 max_score = score
#                 max_label = label
#                 m = idx
#         if len(results["boxes"]) != 0:
#             box = [round(i, 2) for i in results["boxes"][m].tolist()]
#             # calculate metrics
#             ious.append(calculate_iou(annotation["bbox"], box))
#             distances.append(calculate_center_distance(annotation["bbox"], box))
#             if results['labels'][m].item() == 16:
#                 predict_label =  'penguin'
#             else:
#                 predict_label = 'turtle'
#             print(
#                 f"label: {predict_label} ,confidence: "
#                 f"{round(max_score.item(), 3)} , location: {box}"
#             )
#             # 为预测边界框创建一个矩形补丁
#             rect_pred = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
#                                           edgecolor='b', facecolor='none')
#             ax.add_patch(rect_pred)
#             # print(type(max_label))
#             # max_label = max_label.numpy().tolist()[0]
#             if max_label == 16:
#                 max_label = 1
#             else:
#                 max_label = 2
#             # max
#             predicted_labels.append(max_label)
#
#     output_file = os.path.join(output_dir, f'image_{i}.png')
#     plt.savefig(output_file)
#     plt.close()
#
# print('Average IOU:', np.mean(ious))
# print('Standard Deviation of IOU:', np.std(ious))
# print('Average Distance:', np.mean(distances))
# print('Standard Deviation of Distance:', np.std(distances))
#
#
# print(true_labels)
# print(predicted_labels)

# cm = confusion_matrix(true_labels, predicted_labels)
#

# accuracy = accuracy_score(true_labels, predicted_labels)
# precision_penguin = precision_score(true_labels, predicted_labels, pos_label=1)
# recall_penguin = recall_score(true_labels, predicted_labels, pos_label=1)
# f1_penguin = f1_score(true_labels, predicted_labels, pos_label=1)
#
# precision_turtle = precision_score(true_labels, predicted_labels, pos_label=2)
# recall_turtle = recall_score(true_labels, predicted_labels, pos_label=2)
# f1_turtle = f1_score(true_labels, predicted_labels, pos_label=2)
#
# print('Accuracy:', accuracy)
# print('Precision for penguins:', precision_penguin)
# print('Recall for penguins:', recall_penguin)
# print('F1-score for penguins:', f1_penguin)
# print('Precision for turtles:', precision_turtle)
# print('Recall for turtles:', recall_turtle)
# print('F1-score for turtles:', f1_turtle)
#

# print("Confusion Matrix:")
# print(cm)
#

# df_cm = pd.DataFrame(cm,
#   index = ['Actual Penguin', 'Actual Turtle'],
#   columns = ['Predicted Penguin', 'Predicted Turtle']
# )
#

# import seaborn as sns
# plt.figure(figsize = (10,7))
# sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
# plt.show()

# model.save_pretrained("train_outmodel")

