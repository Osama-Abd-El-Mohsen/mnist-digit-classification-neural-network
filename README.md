# 🔢 MNIST Digit Classification with Neural Networks

A deep learning project featuring EDA, model training, and an interactive web application for handwritten digit recognition.

---

## 📑 Table of Contents

- [🔢 MNIST Digit Classification with Neural Networks](#-mnist-digit-classification-with-neural-networks)
  - [📑 Table of Contents](#-table-of-contents)
  - [🎨 Features](#-features)
  - [🚀 Quick Start](#-quick-start)
  - [📱 Usage](#-usage)
  - [📊 Model Architecture](#-model-architecture)
  - [📁 Project Structure](#-project-structure)
  - [🛠 Tech Stack](#-tech-stack)
  - [👤 Author](#-author)
    - [Feedback \& Contributions](#feedback--contributions)

---

## 🎨 Features

| Feature | Description |
|---------|-------------|
| Interactive Canvas | Draw digits directly in browser |
| Real-time Prediction | Instant AI predictions with confidence scores |
| Modern UI | Neon-themed design with smooth animations |
| CNN Model | 99.54 % accuracy on MNIST test set |
| Data Augmentation | Rotation, zoom, and shift transformations |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd app && pip install -r requirements.txt

# 2. Train the model (run notebook)
cd ../notebooks && jupyter notebook EDA_Modeling.ipynb

# 3. Launch the app
cd ../app && streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📱 Usage

1. **Draw** a digit (0-9) on the canvas
2. **Click "Predict"** to get the result
3. **View** confidence score and probability distribution


---

## 📊 Model Architecture

**CNN with Data Augmentation (Production)**

| Layer | Details |
|-------|---------|
| Block 1 | Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(25%) |
| Block 2 | Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(25%) |
| Output | Flatten → Dense(128) → BatchNorm → Dropout(50%) → Softmax(10) |

**Data Augmentation:** Rotation (±10°), Zoom (±10%), Width/Height Shift (±10%)

---

## 📁 Project Structure

```
├── app/
│   ├── app.py              # Streamlit entry point
│   ├── components.py       # UI components
│   ├── prediction.py       # Model prediction logic
│   ├── styles.py           # CSS theming
│   ├── runtime.txt         # runtime version
│   └── requirements.txt    # Dependencies
├── model/
│   └── mnist_model.pkl     # Trained model
├── notebooks/
│   ├── EDA_Modeling.ipynb  # Training pipeline
│   └── utils.py            # Helper functions
└── assets/                 # Static files
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| ML Framework | TensorFlow/Keras |
| Web App | Streamlit |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Dataset | MNIST (70,000 images, 28×28 grayscale) |

**Requirements:** Python 3.8+, TensorFlow 2.x — see [requirements.txt](app/requirements.txt)

---

## 👤 Author
- 🔗 **Portfolio**: [osama-abd-elmohsen-portfolio.me](https://www.osama-abd-elmohsen-portfolio.me)
- 💻 **GitHub**: [@Osama-Abd-El-Mohsen](https://github.com/Osama-Abd-El-Mohsen)

### Feedback & Contributions
Found issues or have suggestions? Contributions and feedback are welcome!


---

**Happy predicting!** 🎉
