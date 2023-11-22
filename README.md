# Heritage Identification Of Monuments Using Deep Learning Techniques

## Overview

This project aims to leverage deep learning techniques to identify and classify heritage monuments from satellite images. We employ the ResNet50 model for the classification task, an Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) for upscaling satellite images, and a comprehensive image preprocessing pipeline for optimal model input.

## Model Used

### ResNet50

We utilize the ResNet50 architecture, a powerful deep learning model known for its exceptional performance in image classification tasks. ResNet50 has demonstrated state-of-the-art results and is well-suited for our goal of accurately classifying heritage monuments.

## Model Pipeline

1. **Image Preprocessing:**
   - **Data Augmentation:** To enhance model robustness, we apply data augmentation techniques such as rotation, scaling, and flipping to the satellite images. This helps the model generalize better to variations in the input data.
   - **Normalization:** Input images are normalized to ensure consistent pixel values, aiding in faster convergence during model training.

2. **ESRGAN for Upscaling Satellite Images:**
   We integrate the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) into our pipeline to enhance the resolution of satellite images. This step improves the clarity and detail of the images, providing a more comprehensive input for the subsequent classification model.

3. **ResNet50 for Heritage Monument Identification:**
   The preprocessed and upscaled satellite images are then passed through the ResNet50 classification model. This deep neural network excels in recognizing patterns and features within images, enabling the accurate identification and classification of heritage monuments.

## Key Features

- **High-Resolution Analysis:** ESRGAN enhances the resolution of satellite images, facilitating detailed analysis for monument identification.
- **Deep Learning Accuracy:** ResNet50, a deep convolutional neural network, ensures accurate and reliable classification of heritage sites.
- **Comprehensive Image Preprocessing:** Data augmentation and normalization contribute to improved model generalization and faster convergence.
- **End-to-End Solution:** The combined pipeline provides an end-to-end solution for heritage identification, from image preprocessing to monument classification.
