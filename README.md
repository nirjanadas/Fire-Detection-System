# 🔥 Real-Time Fire Detection System

## 📌 Project Overview
This project implements a deep learning-based fire detection system using the MobileNet architecture. The model is designed to identify fire in images or video frames in real time, enabling early detection and faster response to potential hazards.

The solution leverages lightweight convolutional neural networks to ensure efficient performance, making it suitable for deployment on resource-constrained devices such as edge systems and mobile platforms.

---

## 🎯 Problem Statement
Fire accidents cause significant loss to life, property, and the environment. Traditional fire detection systems rely on sensors (smoke/temperature), which often:
- Detect fire at a later stage  
- Fail in open or large environments  
- Lack visual confirmation  

This project addresses these limitations using computer vision and deep learning for **early visual fire detection**.

---

## 🚀 Objectives
- Build a fire detection model using MobileNet  
- Classify images into **Fire / No Fire**  
- Enable real-time detection using video/webcam  
- Optimize model for fast inference and low resource usage  

---

## 🧠 Model & Approach

### 🔹 Why MobileNet?
- Lightweight and efficient CNN architecture  
- Uses depthwise separable convolutions  
- Suitable for real-time and edge deployment  

MobileNet-based systems are widely used for fire detection due to their efficiency and ability to run on low-power devices while maintaining high accuracy :contentReference[oaicite:0]{index=0}.

---

### 🔹 Workflow
1. Data Collection (Fire & Non-Fire images)  
2. Data Preprocessing (Resizing, normalization)  
3. Model Training using MobileNet  
4. Evaluation (Accuracy, loss)  
5. Inference on images / video  

---

## 🛠️ Tech Stack
- **Python** – Core programming  
- **TensorFlow / Keras** – Model building  
- **OpenCV** – Image & video processing  
- **NumPy & Pandas** – Data handling  
- **Matplotlib** – Visualization  

---

## 📂 Project Structure
```bash
fire-detection-mobilenet/
│
├── dataset/ # Training & testing images
├── notebooks/ # Model training notebook
├── model/ # Saved trained model
├── scripts/ # Helper scripts
├── app.py / main.py # Entry point for detection
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 🔹 Training
- Load dataset  
- Apply preprocessing  
- Use pre-trained MobileNet (Transfer Learning)  
- Fine-tune model for fire classification  

### 🔹 Prediction
- Input: Image / Video frame  
- Output: Fire / No Fire classification  
- Confidence score  

---

## 📊 Results
- Achieved high accuracy in detecting fire images  
- Efficient inference suitable for real-time applications  
- Reduced computational cost due to lightweight architecture  

---

## 💡 Applications
- Forest fire detection  
- Industrial safety monitoring  
- Smart surveillance systems  
- IoT-based fire alert systems  

---

## ⚡ Key Features
- Real-time fire detection  
- Lightweight and efficient model  
- Scalable for edge deployment  
- Easy integration with camera systems  

---

## 🔮 Future Improvements
- Add smoke detection (multi-class classification)  
- Deploy using Flask / FastAPI  
- Integrate with IoT alert systems  
- Improve dataset diversity for better generalization  

---

## 🧠 Skills Demonstrated
- Deep Learning (CNN, Transfer Learning)  
- Computer Vision  
- Model Optimization  
- Real-time Detection Systems  
- Python Development  

---

## 👤 Author
Nirjana Das
