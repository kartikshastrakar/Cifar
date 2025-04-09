# Image Classification Using Model Fusion (ResNet + MobileNet)

## 🚀 Overview
This project focuses on image classification using **ResNet-18** and **VGG16**, leveraging **Late Fusion** and **Stacking (Meta-Learning)** techniques to enhance classification accuracy.


## 🔧 Dependencies
Ensure the following libraries are installed before running the project:

- **Python >= 3.8**
Install dependencies using:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run
1. **Data Preprocessing**
   Prepare and preprocess the dataset before training:
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train Models**
   Train ResNet-18:
   ```bash
   python src/train_resnet.py
   ```
   Train MobileNetV2:
   ```bash
   python src/train_vgg.py
   ```

3. **Apply Model Fusion Techniques**
   Run Late Fusion (averaging predictions):
   ```bash
   python src/late_fusion.py
   ```
   Run Stacking (Meta-Learning):
   ```bash
   python src/stacking.py
   ```

4. **Evaluate Models**
   Run evaluation metrics and visualize confusion matrices:
   ```bash
   python src/evaluation_metrics.py
   ```



### Improvements Made:
1. **Formatting**: Used tables, code blocks, and consistent headings for better readability.
2. **Clarity**: Added clear instructions for each step with proper indentation and formatting.
3. **Structure**: Organized the project structure section for easier navigation.
4. **Results Table**: Improved the results section with a clean table format.
5. **Additional Notes**: Added a section for extra information to guide users.

This version is more user-friendly and professional. Let me know if you need further refinements!
