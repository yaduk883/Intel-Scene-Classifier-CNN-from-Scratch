This project covers the full deep-learning workflow for image classification: data exploration, preprocessing and augmentation, CNN design and training, and evaluation with per-class metrics and error analysis.

Dataset: Intel Image Classification (Kaggle): about 17,000 labelled 150×150 images across 6 classes (14,034 train, 3,000 test).

Data understanding: class distribution, image-size check, sample grids.
Preprocessing: resize to 150×150, rescale pixels to [0, 1], split seg_train 80/20 into train/validation, use seg_test as the held-out test set.
Augmentation (training only): random horizontal flip, rotation, zoom and translation.
Model: 4 blocks of 2 Conv layers (32, 64, 128, 256 filters) with BatchNorm, ReLU, MaxPooling and Dropout, followed by GlobalAveragePooling, Dense(256), Dropout(0.5) and a softmax output. About 1.24M parameters.
Training: Adam optimizer, with EarlyStopping, ModelCheckpoint and ReduceLROnPlateau callbacks.
Evaluation: accuracy, per-class precision/recall/F1, confusion matrix, and predicted vs. actual visualizations.
Results (test set, 3,000 images)
Class	Precision	Recall	F1
buildings	0.701	0.897	0.787
forest	0.971	0.971	0.971
glacier	0.759	0.741	0.750
mountain	0.748	0.796	0.771
sea	0.863	0.778	0.819
street	0.914	0.745	0.821
Overall accuracy			0.817
Macro avg	0.826	0.821	0.820

#findings

-Forest is the easiest class (F1 0.97).
-The main confusions are street → buildings (110 images), glacier → mountain (104) and mountain → glacier (69), all classes with overlapping visual content.
-The model is somewhat under-trained and unstable rather than overfit, so a smoother LR schedule or transfer learning should improve results.

Libraries Used
Python · TensorFlow/Keras · scikit-learn · Matplotlib · Pillow

Run This on Google Colab - https://colab.research.google.com/drive/1VKL5HlPceQkEQlXSZPp0S7eQkU-AlmnC?usp=sharing
