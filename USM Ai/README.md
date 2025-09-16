# 🎸 USM AI - Universal Smart Musician | Chord Recognition Intelligence

USM AI is an advanced AI-powered system that automatically predicts musical chords from audio tracks, built for musicians, producers, and learners who want deep insights into music harmony. Trained using Beatles albums (*Help!*, *Rubber Soul*, *Revolver*, *Abbey Road*), it extracts multi-dimensional features and maps complex music into a simplified chord vocabulary.

---

## 🚀 Features

- 🎵 **Automatic Chord Recognition**: Predicts chords at each frame of an audio file.
- ⚒️ **Rich Feature Extraction**: Chroma, MFCCs, Spectral Contrast, Tonnetz.
- 🧠 **Deep Learning Model**: CNN + TimeDistributed LSTM architecture.
- 🌿 **Chord Simplification**: Converts raw chord labels into a manageable set (25-30 chords).
- ⚖️ **Balanced Training**: Dynamic class weights to tackle chord imbalance.
- 📂 **Processed Datasets**: Features and labels pre-aligned and ready.
- 🔄 **Scalable Structure**: Modularized for future real-time and mobile deployment.

---

## 📊 Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** (Deep learning)
- **Librosa** (Audio feature extraction)
- **mirdata** (Music datasets)
- **scikit-learn** (Preprocessing)
- **NumPy / pandas / Matplotlib** (Data manipulation & visualization)

---

## 📂 Folder Structure

```
/audio/              # Raw .wav files by album
/annotations/        # .lab chord annotation files
/processed_data/     # Extracted .npy feature and label arrays
/notebooks/          # Jupyter notebooks for exploration and training
/scripts/            # Python scripts for processing, training, evaluation
/models/             # Saved trained model checkpoints
```

---

## 📈 Project Workflow

1. **Extract Features**: Chroma, MFCCs, Spectral Contrast, Tonnetz per frame.
2. **Simplify Chords**: Map detailed chords to basic classes.
3. **Align Labels**: Framewise chord labeling based on annotations.
4. **Train Model**: CNN + TimeDistributed LSTM.
5. **Evaluate**: Metrics like framewise accuracy.
6. **Save Processed Data**: For future fast training.

---

## 🚨 Future Roadmap

- 🔗 Real-Time API for audio upload and live chord prediction (FastAPI backend)
- 📱 Web + Mobile apps for chord visualization
- 📈 Expand training to Isophonics, RWC Pop datasets
- 🔄 Fine-tune on multi-instrument mixtures

---

## 📅 Milestones Completed

- [x] Full feature extraction pipeline
- [x] Feature/label alignment
- [x] Class imbalance handled
- [x] Baseline CNN+LSTM model trained
- [x] Saved processed data for easy reusability

---

## 📊 How to Run

```bash
# Clone the repo
$ git clone https://github.com/yourusername/usm-ai.git

# Setup environment
$ conda create -n usm-ai python=3.10
$ conda activate usm-ai
$ pip install -r requirements.txt

# Extract features
$ python scripts/extract_features.py

# Train model
$ python scripts/train_model.py
```

---

## 💬 Contact

Built with ❤️ by **Akshit Upadhyay**. 
Feel free to connect if you'd like to contribute or collaborate!

- [LinkedIn](https://linkedin.com/in/your-profile)
- [Email](mailto:your.email@example.com)

---

> "Because music deserves to be understood as beautifully as it sounds." 🎵
