# AI Video Sentiment Model - Backend

Backend service for multimodal video sentiment and emotion analysis using:

- Video features
- Audio features
- Speech transcription using Whisper
- Deep learning fusion model

This backend processes uploaded videos and returns:

- utterance-wise transcription
- emotion prediction
- sentiment prediction

---

# Backend Structure

```text
backend/
│
├── deployment/
│   ├── deploy_endpoint.py
│   ├── inference.py
│   ├── models.py
│   ├── requirements.txt
│
├── training/
│   ├── models.py
│   ├── meld_dataset.py
│   ├── train.py
│   ├── requirements.txt
│
├── dataset/
│   └── README.txt
│
├── train_sagemaker.py
├── README.md
└── .gitignore
```

---

# Important Notice

The following files are NOT uploaded to GitHub intentionally:

```text
model.pth
model.tar.gz
videos
venv
node_modules
large datasets
```

These files are excluded because:

- GitHub file size limits
- Large ML models slow repository cloning
- Deployment artifacts should not be version controlled

---

# Required Files Before Running

Place trained model file here:

```text
deployment/model/model.pth
```

Download the trained model from:

```text
https://drive.google.com/file/d/1K1Rfyh8uLAW2W59ottM_0PyLvjRD4byy/view?usp=sharing
```

---

# Backend Setup

## 1. Clone Repository

```bash
git clone <repository-url>
cd ai-video-sentiment-model-main
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate environment:

```bash
.\venv\Scripts\activate
```

---

## 3. Install Backend Dependencies

Navigate to deployment folder:

```bash
cd deployment
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

# FFmpeg Installation

FFmpeg is required for:

- audio extraction
- video segmentation
- Whisper processing

## Windows Setup

Download:

https://www.gyan.dev/ffmpeg/builds/

Steps:

1. Download release build
2. Extract zip
3. Add ffmpeg/bin to system PATH
4. Restart terminal

Verify installation:

```bash
ffmpeg -version
```

---

# Run Local Inference

Inside deployment folder:

```bash
python inference.py
```

The system will:

1. Load trained model
2. Load Whisper transcription model
3. Extract utterances from video
4. Process video frames
5. Process audio features
6. Generate emotion and sentiment predictions

---

# Example Output

```text
0.0s → 6.82s
Text: Tell me about yourself
Emotion: sadness
Sentiment: neutral
```

---

# Backend Pipeline

```text
Input Video
      ↓
Whisper Speech Transcription
      ↓
Utterance Segmentation
      ↓
Video Frame Extraction
      ↓
Audio Feature Extraction
      ↓
Multimodal Deep Learning Fusion
      ↓
Emotion + Sentiment Prediction
```

---

# Main Technologies Used

## Deep Learning

- PyTorch
- Transformers
- Whisper
- Torchvision
- Torchaudio

## Video & Audio Processing

- OpenCV
- FFmpeg

## Deployment

- SageMaker experimentation
- FastAPI integration (local backend)

---

# Team Collaboration Workflow

Before starting work:

```bash
git pull origin main
```

Create a new branch:

```bash
git checkout -b feature-name
```

Commit changes:

```bash
git add .
git commit -m "Added feature"
git push origin feature-name
```

---

# Important Notes

- Keep demo videos short for faster inference ( 1 - 2 mins )
- First Whisper model load may take time
- CPU inference is slower than GPU
- Do NOT commit trained models or videos to GitHub

---

# Contributors

- Backend & ML Pipeline
- Frontend Integration
- Documentation & Testing

---
