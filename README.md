# Pneumonia Detection from Chest X-Rays

> Started as a project for **CS 7375: Artificial Intelligence** (graduate course), Kennesaw State University, Fall 2024, then extended afterward with transfer learning, class balancing, and a web app.

A deep learning model that classifies chest X-rays as Normal or Pneumonia using DenseNet-121 transfer learning. It reaches about 88% test accuracy and 0.95 ROC-AUC, with balanced sensitivity and specificity. Comes with a Streamlit web app for uploading an X-ray and getting a live prediction.

## The problem

Pneumonia is diagnosed largely from chest X-rays, and where trained radiologists are scarce, reading those images is a bottleneck. A model that flags likely pneumonia cases can help triage and speed up diagnosis. The task is binary: given a chest X-ray, predict Normal or Pneumonia.

The data is the Kaggle Chest X-Ray Images (Pneumonia) dataset (Kermany et al.) — 5,856 labeled images, split into training, validation, and test sets.

## Approach: DenseNet-121 transfer learning

The model uses transfer learning from DenseNet-121 pretrained on ImageNet, rather than a network trained from scratch. Images are resized to 224x224 RGB and passed through DenseNet's standard preprocessing. The pretrained backbone acts as a feature extractor, with a new classification head trained on top. To handle the dataset's class imbalance (pneumonia images outnumber normal roughly 2.7 to 1), the model is trained with balanced class weights, so the rarer "normal" class is not ignored. Training uses the Adam optimizer, binary cross-entropy loss, early stopping, and a learning-rate scheduler, with ROC-AUC, precision, and recall tracked alongside accuracy.

## Why transfer learning instead of a CNN from scratch?

This project began as a CNN built from scratch for the course. That version reached 86% accuracy but overfit the small dataset — strong on training data, shaky on validation, and only 55% recall on normal cases, meaning it flagged nearly half of healthy lungs as sick. Switching to DenseNet-121 transfer learning fixed this, for three reasons:

The dataset is small. Under 6,000 images is little for deep learning. A from-scratch network has to learn everything — edges, textures, shapes, and the disease pattern — from those images alone, which tends to cause memorization rather than generalization. DenseNet-121 was pretrained on over a million ImageNet images, so it already knows generic visual features and only needs to adapt them to X-rays.

The results are better and more balanced. With class weighting, normal-case recall rose from 0.55 to 0.81 while pneumonia recall stayed at 0.93, and the model reached 0.95 ROC-AUC — a far more trustworthy classifier for screening.

DenseNet-121 is the right architecture for this task. It is the same architecture as CheXNet, the model that demonstrated radiologist-level pneumonia detection on chest X-rays. Choosing it was a deliberate, literature-informed decision, not a default.

A from-scratch CNN is not wrong — it offers full control and is good for understanding how convolutional networks work, which made it a reasonable first version. The trade-off is that it needs far more data and compute to compete, and transfer learning is the better fit when data is limited, as it usually is in medical imaging.

## Results

On the held-out test set: about 88% accuracy and 0.95 ROC-AUC.

| Metric | Value |
|--------|-------|
| Test accuracy | ~88% |
| ROC-AUC | 0.95 |
| Recall (Pneumonia) | 0.93 |
| Recall (Normal) | 0.81 |
| Precision (both classes) | ~0.88-0.89 |

The model catches most pneumonia cases while keeping false alarms on healthy lungs in check — the balance you want from a screening tool, where missing a real case is costlier than a false positive that gets a second look.

## Web app

The repo includes a Streamlit app that loads the trained model and returns a live prediction with a confidence score. Once the trained model file is in place (see below), launch it with:

```bash
streamlit run app.py
```

It opens in your browser. Upload a chest X-ray and it returns Normal or Pneumonia, the confidence, and the raw model probability. The app carries a clear disclaimer that it is a learning project, not a diagnostic tool.

## Run it yourself

The model is trained in a notebook (Kaggle or Google Colab with a free GPU). After training, save the model as `pneumonia_model.h5`, download it, and place it in the `src/` folder. Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The X-ray dataset is not included here (it is large and distributed under its own terms). Download Chest X-Ray Images (Pneumonia) from Kaggle to retrain: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## What I'd improve

Add Grad-CAM heatmaps so the app shows where in the X-ray the model is looking, which builds trust in a medical setting. Try fine-tuning the deeper DenseNet layers for a possible further gain, and deploy the app to Streamlit Community Cloud for a public live demo.

## Notes

The dataset is the public Kaggle Chest X-Ray Images (Pneumonia) set (Kermany et al.), not redistributed here.
