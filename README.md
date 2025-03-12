
# Heart Image Classification Using Neural Networks

This project leverages machine learning to classify heart images into four categories based on pathology: Heart failure with infarction (HF-I), Heart failure without infarction (HF), Hypertrophy (HYP), and Normal (N). The classification model is built using TensorFlow, pre-processed images in DICOM format, and patient data in CSV format. The model has been containerized into an app using FlaskAPI and Docker.

## Requirements
Before running the code, make sure the following libraries are installed:

```bash
pip install tensorflow
pip install pydicom
pip install Pillow
pip install seaborn
pip install opencv-python
```

## Project Overview
The project involves several key steps:

### Data Extraction and Preprocessing:
- The raw image files in DICOM format are unzipped and converted to JPEG format.
- The images are resized and categorized into one of four pathologies based on a supplemental CSV file containing patient data.

### Image Preprocessing:
- The DICOM files are read and resized to 224x224 pixels.
- Each image is associated with a pathology label and saved in the corresponding folder.

### Neural Network Model:
- A convolutional neural network (CNN) is built to classify the heart images into one of four categories using the processed data.

### Model Training and Evaluation:
- The model is trained on the image data, and its performance is evaluated using accuracy metrics.
- A test set is used for evaluating model performance after training.

### Saving and Predicting:
- The trained model is saved to Google Drive for future use.
- The model is used to make predictions on new test data, and the results are displayed.

### Flask API and Docker Containerization:
- The model has been integrated into a FlaskAPI app, which allows for prediction requests via API calls.
- The app is containerized using Docker to make deployment easier and more portable. The repository includes:
  - **app.py**: The Flask API app file.
  - **Dockerfile**: The Dockerfile for containerization.
  - **requirements.txt**: The list of Python dependencies.
  - **model_script.py**: The script used to create and train the models.

## Project Setup

1. **Unzip Image Files**: Ensure you have access to the DICOM image files and unzip them using the following commands:
   
   ```bash
   !unzip /content/SCD_IMAGES_05.zip
   !unzip /content/SCD_IMAGES_01.zip
   !unzip /content/SCD_IMAGES_03.zip
   ```

2. **Preprocess Data**: A CSV file (`scd_patientdata.csv`) containing patient data with the following columns is required:
   - **PatientID**: Unique identifier for each patient.
   - **Pathology**: The pathology label associated with the patient.
   
   This data is used to categorize the images into one of the four classes.

3. **Process DICOM Images**: The `process_dicom` function processes the DICOM files, resizing them and saving them in JPEG format. It also assigns the correct labels to each image.

4. **Train Neural Network**: The model consists of several convolutional layers followed by dense layers, using the ReLU activation function and softmax for multi-class classification. The model is trained on the processed images.

5. **Evaluate the Model**: After training, the model’s performance is evaluated on the test set, and the results (accuracy) are displayed.

6. **Save the Model**: The model is saved to Google Drive for future inference:

   ```python
   model.save('/content/drive/MyDrive/heart_model.h5')
   ```

7. **Make Predictions**: Use the trained model to make predictions on the test images. The predicted and true labels for the images are displayed.

## Example Output
When running the model, you will see the following outputs:
- Number of images processed per category.
- The model’s training and validation accuracy.
- Example image predictions alongside the true labels.

```bash
Test Loss: 0.3157
Test Accuracy: 87.5%
```

The predicted labels and true labels are shown alongside images in a plot.

## Notes
- The model is trained with 10,000 images sampled from each class (with a reduced sample size per class to avoid overfitting).
- Ensure that the correct file paths for the data and images are set when running the code.
- You can adjust the number of epochs, batch size, and image sample size according to your system's capabilities.

## Data Resource
[Sunnybrook Cardiac Data](https://www.cardiacatlas.org/sunnybrook-cardiac-data/)
