import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import json

EMOTION_MAP = {
    0: "anger", 1: "disgust", 2: "fear",
    3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"
}

SENTIMENT_MAP = {
    0: "negative", 1: "neutral", 2: "positive"
}


# ---------------- VIDEO PROCESSOR ----------------
class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        while len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted")

        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


# ---------------- AUDIO PROCESSOR ----------------
class AudioProcessor:
    def extract_features(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")

        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        waveform, sr = torchaudio.load(audio_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64
        )(waveform)

        mel = (mel - mel.mean()) / mel.std()

        if mel.size(2) < 300:
            mel = torch.nn.functional.pad(mel, (0, 300 - mel.size(2)))
        else:
            mel = mel[:, :, :300]

        os.remove(audio_path)

        return mel


# ---------------- SEGMENT EXTRACTOR ----------------
class SegmentProcessor:
    def extract_segment(self, video_path, start, end):
        segment_path = f"segment_{start}_{end}.mp4"

        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return segment_path


# ---------------- MODEL LOAD ----------------
def model_fn(model_dir="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalSentimentModel().to(device)

    model_path = os.path.join(model_dir, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Loading Whisper model...")
    transcriber = whisper.load_model("base", device=device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "transcriber": transcriber,
        "device": device
    }


# ---------------- PREDICT ----------------
def predict_fn(video_path, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    transcriber = model_dict["transcriber"]
    device = model_dict["device"]

    print("Running Whisper transcription...")
    result = transcriber.transcribe(video_path, word_timestamps=True)

    video_proc = VideoProcessor()
    audio_proc = AudioProcessor()
    segment_proc = SegmentProcessor()

    outputs = []

    for seg in result["segments"]:
        try:
            segment_path = segment_proc.extract_segment(
                video_path, seg["start"], seg["end"]
            )

            video = video_proc.process_video(segment_path)
            audio = audio_proc.extract_features(segment_path)

            text = tokenizer(
                seg["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            text = {k: v.to(device) for k, v in text.items()}
            video = video.unsqueeze(0).to(device)
            audio = audio.unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(text, video, audio)

                emo_probs = torch.softmax(out["emotions"], dim=1)[0]
                sen_probs = torch.softmax(out["sentiments"], dim=1)[0]

            outputs.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "emotion": EMOTION_MAP[emo_probs.argmax().item()],
                "sentiment": SENTIMENT_MAP[sen_probs.argmax().item()]
            })

        except Exception as e:
            print("Segment failed:", e)

        finally:
            if os.path.exists(segment_path):
                os.remove(segment_path)

    return outputs


# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":
    video_path = "./sample_video.mp4"

    model_dict = model_fn("model")  # ✅ FIXED PATH

    results = predict_fn(video_path, model_dict)

    for r in results:
        print("\n-------------------")
        print(f"{r['start']}s → {r['end']}s")
        print("Text:", r["text"])
        print("Emotion:", r["emotion"])
        print("Sentiment:", r["sentiment"])
