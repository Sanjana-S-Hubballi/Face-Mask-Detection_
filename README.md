# Face Mask Detection using Mobilenetv2

## Overview
This project aims to detect and classify faces in an image as either masked or unmasked.

## Workflow
- Resnet10 model is used to detect the faces from the dataset, and the detected faces are saved in a separate folder.
- Re-train the mobilenetv2 model on traindataset.
- The traindataset images are preprocessed and passed through an image augmentation pipeline to improve model generalization. This includes transformations such as:
  - Random horizontal flips
  - Rotation and zoom
  - Brightness and shear adjustments
  - Shifts in width and height: This augmentation ensures the model can handle various orientations, lighting conditions, and facial variations.
- Custom classification head is added, which includes a global average pooling layer, a dropout layer to reduce overfitting, and a final dense layer with a sigmoid activation for binary classification (mask vs nomask).
-  Feature extraction: In the first training phase, the base MobileNetV2 layers are frozen. Only the new classification layers are trained. This allows the model to quickly learn how to classify masks using the already learned image features.
-  After the initial training, the last few layers of MobileNetV2 are unfreezed and trained again at a lower learning rate. This fine-tuning step allows the pre-trained weights to adjust slightly to better fit our specific face mask dataset — improving performance without overfitting.

## Training Summary
| **Metric**             | **Value**  |
|------------------------|------------|
| **Training Accuracy**  | 91.44%     |
| **Validation Accuracy**| 91.72%     |
| **Training Loss**      | 23.82%     |
| **Validation Loss**    | 22.84%     |

## Confusion matrix
![confusion_matrix](https://github.com/user-attachments/assets/e4e50085-1581-46cf-9981-02674aaa32e6)

### False Acceptance Rate and False Rejection Rate
![FAR_FRR](https://github.com/user-attachments/assets/09573f15-83a8-4f39-9241-ec4981d65892)

