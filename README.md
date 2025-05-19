
# 🎵 **Chorus.AI**

*A GenAI-driven music composition tool powered by deep learning and Intel® Tiber Cloud.*

---

## 🚀 Features

* 🎼 **Generates music** based on learned patterns and audio features
* 🧠 **Neural audio synthesis** using deep generative models
* 🎛️ **Feature-controlled input** (e.g., pitch, instrument, timbre)
* 🎧 **Model training and inference** for high-quality audio generation
* 💾 **Integrated datasets**: NSynth, NES-MDB, and custom inputs
* 🕹️ **Real-time music generation** via Streamlit UI + interactive 3D visualizer (Three.js)
* 🎹 **MIDI input/output** support for live composition
* 🧬 **Genre conditioning** and control over style
* ☁️ **Optimized and  on Intel® Tiber Developer Cloud**

---

## 🧰 Tech Stack

* 🐍 Python
* 🔶 TensorFlow / 🟣 PyTorch
* 🎚️ Librosa / Torchaudio
* 🎵 NSynth 
* 📓 Jupyter Notebooks (for experimentation)
* 🌐 Streamlit (interactive UI)
* 🧊 Three.js (3D visual visualizer integration)
* ☁️ Intel® Tiber Developer Cloud

---

## 📦 Installation

1. 📥 Clone the repository:

```bash
git clone https://github.com/yourusername/music-generator.git
cd music-generator
```

2. 🛠️ Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 Dataset

This project supports:

* 🎶 [NSynth](https://magenta.tensorflow.org/datasets/nsynth)


📁 Preprocessing scripts are provided under the `data/` directory.

---

## 🏁 Usage

### 1. 🧹 Preprocess Data

```bash
python scripts/preprocess_nsynth.py
```

### 2. 🏋️ Train the Model

```bash
python scripts/train_model.py --dataset nsynth
```

✨ Training is accelerated using Intel® Tiber Developer Cloud.

### 3. 🎼 Generate Music

```bash
python scripts/generate.py --model models/latest.pth
```

### 4. 🖥️ Launch the Real-Time Interface

```bash
streamlit run ui/app.py
```

Includes:

* 🎛️ Feature sliders for custom generation
* 🎹 Live MIDI input
* 🧊 3D waveform & style visualizer (Three.js)

---


## 🤝 Contributing

Want to improve it? Open an issue or a pull request. Contributions are welcome! 🎉

---

## 📜 License

📝 MIT License. See [LICENSE](./LICENSE) for full terms.

---


