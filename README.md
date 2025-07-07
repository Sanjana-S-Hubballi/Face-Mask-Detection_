# Face Mask Detection using Mobilenetv2

## Overview
This project aims to detect and classify faces in an image as either masked or unmasked.

## Workflow
- First, the faces are detected using Resnet10 model, and the detected faces are saved in a separate folder.
- Classify the faces in the traindataset using Mobilenetv2 model into faces with mask and faces without mask.
- Re-train the Mobilenetv2 model
- 
