# Real-Time Emotion Detection Using CNN and OpenCV

## Project Description
This project implements a **real-time facial emotion recognition system** using a **Convolutional Neural Network (CNN)** trained on the **FER2013 dataset**. It detects faces from a webcam feed, predicts the emotion of each detected face, and displays it live on the video stream.  

The model classifies seven emotions:  
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## Features
- Real-time emotion detection from webcam feed
- Face detection using Haar Cascade classifier
- Data augmentation and normalization for better performance
- CNN model with multiple convolutional and pooling layers
- Displays bounding box and predicted emotion label with confidence score

---

## Requirements
- Python 3.8+
- TensorFlow
- OpenCV
- kagglehub
- NumPy

Install dependencies using pip:

```bash
pip install kagglehub tensorflow opencv-python numpy
```


---

## Dataset
This project uses the **FER2013 dataset** from Kaggle:  
- Downloaded programmatically using `kagglehub`.  
- Dataset is split into `train` and `test` directories.  
- Images are 48x48 pixels, grayscale, and labeled with one of the seven emotions.  

Kaggle Dataset Link: [FER2013](https://www.kaggle.com/msambare/fer2013)

---

## Project Structure
```
emotion-recognition/
│
├── emotion_model.h5 # Trained CNN model
├── Emotion_Detection.ipynb # Real-time emotion detection script
├── README.md # Project description and instructions
└── dataset/ # FER2013 dataset downloaded via kagglehub
```
---

## Steps / Workflow

1. **Install Dependencies**  
   Install required Python libraries (`tensorflow`, `opencv-python`, `kagglehub`, `numpy`).  

2. **Download Dataset**  
   Download FER2013 dataset using `kagglehub`.  

3. **Preprocess Data**  
   - Resize images to 48x48  
   - Normalize pixel values  
   - Apply data augmentation for training (rotation, zoom, horizontal flip)  

4. **Build CNN Model**  
   - 3 convolutional layers with increasing filters (32 → 64 → 128)  
   - MaxPooling layers after each convolution  
   - Flatten, Dense layers, Dropout for regularization  
   - Output layer with 7 neurons (softmax activation)  

5. **Train Model**  
   - Use categorical crossentropy loss and Adam optimizer  
   - Train for 30 epochs (or more for better accuracy)  
   - Validate using test dataset

6. **Save Model**  
   Save the trained model as `emotion_model.h5`.  

7. **Real-Time Detection**  
   - Capture webcam feed using OpenCV  
   - Detect faces using Haar Cascade  
   - Preprocess detected face and predict emotion using trained CNN  
   - Draw bounding box and display emotion with confidence  

8. **Exit**  
   Press `q` to quit the webcam feed.

---

## Usage

1. Make sure `emotion_model.h5` is present in the project folder.  
2. Run the script:

```bash
python Emotion_Detection.ipynb
```

3. Allow access to the webcam.
4. View live emotion detection on detected faces.

---

## Results
- The model achieves a reasonable accuracy on the FER2013 dataset (~65–70% with basic CNN; can be improved using advanced architectures like VGG or ResNet).  
- Real-time webcam detection shows emotion labels with confidence scores above each detected face.

---

## Notes
- For better performance, consider training with more epochs or deeper CNN architectures.  
- GPU acceleration is recommended for faster training and real-time inference.  
- Ensure proper lighting and camera position for accurate detection.










