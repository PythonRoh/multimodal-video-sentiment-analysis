"use client";

import { useState } from "react";
import { FiUpload } from "react-icons/fi";
import type { Analysis } from "./Inference";

interface UploadVideoProps {
  apiKey: string;
  onAnalysis: (analysis: Analysis) => void;
}

function UploadVideo({ apiKey, onAnalysis }: UploadVideoProps) {
  const [status, setStatus] = useState<"idle" | "uploading" | "analyzing">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    try {
      setStatus("uploading");
      setError(null);

      const fileType = `.${file.name.split(".").pop()}`;

      // 1. Get unique key from Vercel
      const res = await fetch("/api/upload-url", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ fileType: fileType }),
      });

      if (!res.ok) {
        const errorData = (await res.json()) as { error?: string };
        throw new Error(errorData?.error ?? "Failed to initialize upload");
      }

      const { key } = (await res.json()) as { key: string };

      // 2. Upload file directly to Hugging Face
      const formData = new FormData();
      formData.append("file", file, key);

      const aiServerUrl = "https://rudraop-video-sentiment-analysis.hf.space/analyze";
      
      const aiRes = await fetch(aiServerUrl, {
        method: "POST",
        body: formData,
      });

      if (!aiRes.ok) {
        throw new Error(`AI Backend Error (${aiRes.status})`);
      }

      setStatus("analyzing");
      const analysisResultRaw = await aiRes.json() as { utterances?: any[]; error?: string; trace?: string };

      if (analysisResultRaw.error) {
        console.error("AI Backend Error Trace:", analysisResultRaw.trace);
        throw new Error(`AI Engine Error: ${analysisResultRaw.error}`);
      }

      const analysisResult = analysisResultRaw as Analysis;

      // 3. Finalize with Vercel
      const confirmRes = await fetch("/api/sentiment-inference", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ key, analysis: analysisResult }),
      });

      if (!confirmRes.ok) {
        throw new Error("Failed to save analysis results");
      }

      onAnalysis(analysisResult);
      setStatus("idle");
    } catch (error) {
      setError(error instanceof Error ? error.message : "Upload failed");
      setStatus("idle");
    }
  };

  return (
    <div className="flex w-full flex-col gap-2">
      <div className="flex w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-gray-300 p-10">
        <input
          type="file"
          accept="video/mp4,video/mov,video/avi"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              handleUpload(file).catch((err) => {
                console.error("Error uploading file:", err);
              });
            }
          }}
          id="video-upload"
        />
        <label
          htmlFor="video-upload"
          className="flex cursor-pointer flex-col items-center"
        >
          <FiUpload className="min-h-8 min-w-8 text-gray-400" />
          <h3 className="text-md mt-2 text-slate-800">
            {status === "uploading"
              ? "Uploading to AI Server..."
              : status === "analyzing"
                ? "AI Processing Video..."
                : "Upload a video"}
          </h3>
          <p className="text-center text-xs text-gray-500">
            {status === "idle" ? "Get started with sentiment detection by uploading a video." : "Please wait while we process your video."}
          </p>
        </label>
      </div>
      {error && <div className="mt-2 text-center text-sm text-red-500">{error}</div>}
    </div>
  );
}

export default UploadVideo;
