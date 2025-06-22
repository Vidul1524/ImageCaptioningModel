# ImageCaptioningModel:
This project implements an Image Captioning Model — a deep learning pipeline that combines Computer Vision and Natural Language Processing (NLP) to generate natural language descriptions for images.

Given an image as input, the model generates a sentence that best describes the scene.

🔍 What’s Inside:
DenseNet201 (pre-trained CNN) to extract meaningful image features

Tokenization & Padding to process textual captions

LSTM & Bidirectional LSTM to model the caption generation

Custom Data Generator for efficient training using (Image, Partial Caption) → Next Word logic

Training Optimizations using callbacks like EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint

✅ Technologies Used:
TensorFlow / Keras

Python

CNN + RNN-based architecture

Functional API in Keras

Matplotlib, NumPy, and Pandas for preprocessing and visualization

📦 Dataset:
The model was trained on the Flickr8k Dataset, which consists of:

8,000 images

5 captions per image
