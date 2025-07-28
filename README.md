# Shoe Image Classification Project

This project implements a deep learning solution for classifying shoe images into three categories: Adidas, Converse, and Nike. It leverages transfer learning with the InceptionV3 convolutional neural network architecture and employs a robust two-stage training strategy to achieve high accuracy.

## Project Overview

The primary objective of this project is to develop an accurate image classification model for differentiating between various shoe brands. The key steps involved are:

1.  **Data Loading and Preprocessing**: Images are loaded from a structured directory and organized into training, validation, and test dataframes. Data augmentation is applied to the training set to improve model generalization.
2.  **Exploratory Data Analysis (EDA)**: Visualizations are generated to understand the distribution of classes within the dataset and across different splits (train, validation, test). Sample images from each class are also displayed.
3.  **Model Architecture**: A pre-trained InceptionV3 model, trained on the ImageNet dataset, is used as the base. Custom dense layers with regularization, LeakyReLU activation, and batch normalization are added on top of the base model for the specific classification task.
4.  **Two-Stage Training Strategy**:
    *   **Stage 1 (Feature Extraction)**: The InceptionV3 base model's layers are frozen, and only the newly added custom layers are trained. This allows the model to learn how to map the extracted features to the target classes without modifying the powerful pre-trained feature extractor. An Adam optimizer with an initial learning rate of 0.001 is used. Callbacks for early stopping, learning rate reduction on plateau, and model checkpointing are employed.
    *   **Stage 2 (Fine-tuning)**: The last 20 layers of the InceptionV3 base model are unfrozen, allowing them to be trained along with the custom layers. This fine-tuning step adapts the pre-trained features to the specific shoe dataset, often leading to improved performance. The model is re-compiled with an Adam optimizer (initial learning rate also set to 0.001 for demonstration in the notebook, but a smaller learning rate is often preferred for fine-tuning) and the same callbacks.
5.  **Model Evaluation**: The trained model is evaluated on the independent test set to assess its generalization performance. Key metrics such as test loss and test accuracy are reported.
6.  **Performance Analysis**: A confusion matrix and a detailed classification report (including precision, recall, and f1-score for each class) are generated to provide a deeper understanding of the model's performance and identify potential areas for improvement.
7.  **Prediction Visualization**: Sample images from the test set are displayed along with their true labels and the model's predicted labels, visually highlighting correct and incorrect predictions.

## Dataset

The dataset is expected to be organized in a directory structure where the base directory (`base_path`) contains subdirectories for 'train', 'test', and 'val'. Within each of these split directories, there should be further subdirectories named after each class (e.g., 'adidas', 'converse', 'nike') containing the corresponding image files.
