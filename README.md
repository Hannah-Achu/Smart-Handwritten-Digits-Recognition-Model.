# Smart Handwritten Digits Recognition

## Introduction
This project implements a **Smart Handwritten Digits Recognition Model**.  
The goal of the project was to build a model capable of identifying handwritten digits (0–9) from images.  

Even though the final model achieved an accuracy of **19%**, the project successfully demonstrates the process of:

- Preparing image data for machine learning
- Building and training a Convolutional Neural Network (CNN)
- Evaluating model performance using common metrics

---

## Tools and Libraries
The project was implemented in **Python** using **Google Colab**. Key tools and libraries used include:

- **Python**: Programming language  
- **Google Colab**: Platform for running the code  
- **OpenCV (cv2)**: Reading, resizing, and converting images to grayscale  
- **NumPy**: Handling arrays and numerical operations  
- **Pandas**: Managing tabular data and saving performance results  
- **scikit-learn**: Splitting dataset, calculating accuracy, confusion matrix, and classification report  
- **Seaborn & Matplotlib**: Visualizing confusion matrices  
- **TensorFlow & Keras**: Building and training the CNN model  
- **Google Drive**: Storing dataset and saving output files  

---

## Methodology

### 1. Loading Libraries
All required libraries were imported at the beginning of the notebook.

### 2. Mounting Google Drive
The dataset was stored in Google Drive and accessed directly from Colab.

### 3. Loading and Preprocessing Images
- Images were read from subfolders labeled **0–9**.  
- All images were converted to **grayscale** and resized uniformly.  
- Images not belonging to any label were temporarily assigned `-1` and excluded from training.

### 4. Preparing Data for Training
- Labeled images were converted to **NumPy arrays**.  
- Pixel values were normalized to the range **[0, 1]**.  
- Data was split into **70% training** and **30% testing** using `train_test_split()`.

### 5. Building the CNN Model
- A **Sequential CNN** was created to extract features from images, reduce spatial dimensions, and learn deeper patterns.  
- The CNN architecture included layers for feature extraction and classification.

### 6. Training the Model
- The model was trained for **10 epochs** with a batch size of 32.  
- Both training and validation accuracy were monitored during the process.

### 7. Evaluating the Model
Model performance was evaluated using:

- **Accuracy Score**  
- **Confusion Matrix**: Visualized using Seaborn heatmap  
- **Classification Report**: Includes precision, recall, and F1-score for each class  

### 8. Saving Results
- Confusion matrix and performance metrics were saved in **Performance.xlsx**, with separate sheets for each.

---

## Results and Observations
- The CNN model achieved **~70% accuracy** on the test data.  
- The relatively low accuracy was expected due to:  
  - Small and unevenly distributed dataset  
  - Some images may not have been clearly labeled or preprocessed uniformly  
  - Limited training epochs (10), insufficient for the model to fully learn patterns  

---

## Conclusion
Despite the low accuracy, this project provided valuable learning experience in:

- Converting images to arrays and normalizing pixel values  
- Understanding how CNNs extract and learn features from image data  
- Evaluating model performance using metrics like accuracy, confusion matrix, and classification report  

---

## How to Run
1. Open the notebook in **Google Colab**.  
2. Mount your Google Drive where the dataset is stored:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

