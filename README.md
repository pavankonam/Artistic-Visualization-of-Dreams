# Artistic Visualization of Dreams Using EEG

## Overview
This project, developed by Pavan Konam and Ashlesha Ahirwadi, aims to decode and reconstruct visual dream content from EEG (Electroencephalography) signals. By leveraging EEG data and advanced machine learning techniques, the project translates brain activity into categorized dream imagery, offering a novel way to visualize the subconscious mind.

## Project Goal
The primary objective is to analyze EEG signals recorded during sleep, classify dream states into five distinct categories, and generate artistic representations of dream content using a combination of machine learning models and image generation tools.

## Methodology
### Data Sources
- **Datasets**: Five diverse datasets (Zhang & Wamsley 2019, Oudiette_N1Data, LODE, TWC_USA, Donders) containing EEG data in EDF format and accompanying dream descriptions in text files.
- **Languages**: English, French, Italian, and conversational English.
- **Synthetic Data**: EEG segments matched with dream content keywords and categorized into five classes.

### EEG Data Preprocessing
- **Feature Extraction**: Highpass and bandpass filters applied to isolate Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), and Gamma (30+ Hz) waves.
- **Cleaning**: Removal of noise and handling of missing values.
- **Features**: Power Spectral Density (PSD) mean/std, variance, skewness, kurtosis.

### Text Data Preprocessing
- Converted text files to CSV format.
- Translated all text to English.
- Extracted keywords and classified them into five categories.
- Combined datasets into a single CSV file.

### Model Development
1. **Baseline Model**: Random Forest
   - Data split: Training (85,521 samples), Validation (21,381 samples), Test (11,878 samples).
   - Validation Accuracy: 88.19%, Test Accuracy: 87.97%.
   - Strengths: Simple, interpretable, and effective for non-linear data.

2. **CNN with Spectrograms**
   - EEG signals transformed into spectrograms using STFT or Wavelet Transform.
   - CNN Architecture: 3 convolutional layers, max pooling, dense layers, and dropout.
   - Training: 30 epochs with Adam optimizer and CrossEntropy loss (approx. 4 hours).
   - Test Accuracy: 87.79%.

### Image Generation
- **Pipeline**: CNN extracts dream features from spectrograms, which are then fed into DALL-E 3 for image generation.
- **Example Category**: "Adventure and Movement" – characterized by dynamic scenes tied to high activity in visual processing and motor imagery frequency bands.

## Results
- Achieved ~88% accuracy in classifying dream states.
- Successfully generated artistic visualizations of dream content, such as adventure-themed imagery.

## Limitations
- Limited EEG resolution and data scarcity.
- Variability across individuals affecting generalization.

## Future Work
- Develop personalized dream decoding models.
- Enable real-time dream visualization systems.

## Repository Contents
- **PDF Document**: Detailed explanation of the project methodology, datasets, and results.
- **Code**: Scripts for EEG preprocessing, model training, and image generation (to be uploaded).
- **Sample Outputs**: Example spectrograms and generated dream images (to be added).

## Dependencies
- Python (for data processing and model development).
- Libraries: NumPy, Pandas, SciPy, TensorFlow/PyTorch (for CNN), and DALL-E 3 API (for image generation).

