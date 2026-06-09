# (Discontinued) Skin Lesion Classifier using HAM10000 

A deep learning project to classify dermatoscopic images of pigmented skin lesions into seven diagnostic categories.  
Built with TensorFlow/Keras and transfer learning, using the [HAM10000 ("Human Against Machine") dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

---

## Motivation
This project was undertaken to:
- Gain hands-on experience with medical image classification.
- Learn how to build a production-like data pipeline using `tf.data`.
- Apply transfer learning with pre-trained models (EfficientNet).
- **Primary goal:** utilise a local GPU for accelerated training on Windows.

---

## Dataset
- **Name:** HAM10000 (Human Against Machine with 10,000 training images)
- **Size:** 10,015 dermatoscopic images of pigmented skin lesions
- **Classes:** 7 diagnostic categories
  - `akiec` – Actinic keratoses / Bowen’s disease
  - `bcc` – Basal cell carcinoma
  - `bkl` – Benign keratosis-like lesions
  - `df` – Dermatofibroma
  - `mel` – Melanoma
  - `nv` – Melanocytic nevi
  - `vasc` – Vascular lesions
- **Metadata:** Includes patient age, sex, lesion location, and verification method.
- **License:** CC BY-NC-SA 4.0

To avoid data leakage, only the first image of each lesion was retained, resulting in ~7,470 unique lesions.

---
# Progress Summary

### 1. Data Exploration (EDA)
- Loaded the metadata and inspected class distribution (highly imbalanced, `nv` dominates with 67%).
- Visualised sample images from each class to understand visual differences.
- Identified duplicate images belonging to the same lesion (`lesion_id` appeared multiple times).
- **Deduplication strategy:** Kept only the first occurrence of each `lesion_id` to prevent data leakage.
- Encoded string labels to integers using `LabelEncoder`.
- Performed a stratified train/validation split (80/20) on the deduplicated dataset.

### 2. Data Pipeline
- Implemented a `tf.data` pipeline with the following steps:
  - **Loading:** `tf.io.read_file` → `tf.image.decode_jpeg`
  - **Preprocessing:** resize to 224×224, normalise pixel values to [0,1]
  - **Augmentation (training only):** random horizontal/vertical flips, 90° rotations, small brightness/contrast adjustments, all clipped to [0,1]
  - **Performance:** shuffling, batching (32), and `prefetch(tf.data.AUTOTUNE)`
- Built a reusable `create_dataset()` function that accepts a DataFrame and a `training` flag.
- Verified the pipeline by displaying a batch of augmented images.

### 3. Model Building
- Chose **EfficientNetB0** pre-trained on ImageNet as the feature extractor.
- Frozen the base model initially (`trainable = False`).
- Added a custom classification head:
  - Lambda layer to apply EfficientNet’s official preprocessing (rescaling from [0,1] to [0,255] then applying `preprocess_input`)
  - Global average pooling
  - Dropout (0.3)
  - Dense layer with 7 outputs and softmax activation
- Computed class weights using `sklearn` to handle extreme class imbalance.
- **Compiled the model** with:
  - Optimizer: Adam (lr=0.001)
  - Loss: Sparse categorical crossentropy
  - Metrics: Accuracy

**Status:** The model architecture is ready for training, but training was not executed.

---

## Reason for Discontinuation
The project was halted at the model compilation stage due to **GPU compatibility issues on native Windows**.

- TensorFlow ≥2.11 **no longer supports GPU acceleration on native Windows** – only CPU execution is available, even when CUDA/cuDNN are installed.
- The alternative `tensorflow-directml` package (which provides GPU support via DirectML) **cannot be installed** on Python 3.12+ because pre-built wheels are not distributed for those versions.
- Since one of the core learning objectives was to utilise the local GPU, the project was **formally discontinued** before training could begin.

The code remains in a fully functional state and can be trained on CPU, or on a GPU within a WSL2/Linux environment.

