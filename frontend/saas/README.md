# Frontend – Multimodal Video Sentiment Analysis

This frontend powers the user interaction layer of the Multimodal Video Sentiment Analysis system. It provides an intuitive SaaS-style interface for uploading videos, handling authentication, processing AI inference requests, and visualizing sentiment analysis results in real time.

The application is built with a modern full-stack frontend architecture using Next.js, TypeScript, and Tailwind CSS.

---

## Overview

The frontend is responsible for:

- User authentication
- Video upload workflows
- Sending inference requests to the backend
- Displaying AI-generated sentiment outputs
- Managing client-side interactions and UI states
- Providing a scalable and modular interface for future AI integrations

---

# Tech Stack

- **Next.js**
- **React**
- **TypeScript**
- **Tailwind CSS**
- **Server Actions**
- **REST API Integration**

---

# Project Structure

```bash
src/
│
├── actions/              # Server actions and authentication handlers
│   └── auth.ts
│
├── app/                  # App router pages and API routes
│   ├── api/
│   ├── login/
│   ├── signup/
│   ├── layout.tsx
│   └── page.tsx
│
├── components/           # Reusable frontend components
│   └── client/
│       ├── UploadVideo.tsx
│       ├── Inference.tsx
│       ├── signup.tsx
│       └── copy-button.tsx
│
├── lib/                  # Utility functions and helpers
├── schemas/              # Validation schemas
├── server/               # Backend communication layer
├── styles/               # Global styles and UI styling
│
├── middleware.ts         # Middleware logic
└── env.js                # Environment configuration
```

---

# Features

## Authentication System

- Login and signup flows
- Session handling
- Route protection using middleware
- Authentication server actions

---

## Video Upload Pipeline

Users can:

- Upload multimedia content
- Trigger AI inference requests
- Preview uploaded content
- Interact with sentiment analysis workflows

---

## AI Inference Interface

The frontend communicates with backend AI services to:

- Send uploaded video/audio data
- Receive processed inference results
- Display emotion and sentiment outputs dynamically

---

## Modular Component Architecture

The UI is structured using reusable client components to maintain:

- Scalability
- Cleaner state management
- Easier maintenance
- Better code separation

---

# Environment Setup

Create a `.env` file in the project root.

Example:

```env
DATABASE_URL=
NEXTAUTH_SECRET=
NEXTAUTH_URL=
API_BASE_URL=
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/PythonRoh/multimodal-video-sentiment-analysis.git
```

Move into frontend directory:

```bash
cd frontend/saas
```

Install dependencies:

```bash
npm install
```

Start development server:

```bash
npm run dev
```

---

# Future Improvements

- Real-time inference streaming
- AI visualization dashboards
- Video timeline sentiment mapping
- Cloud deployment support
- Multi-user collaboration
- Advanced analytics and reporting
- Dark/light theme customization

---

# Contributors

- Rudra
- Rohit Rout

---

# License

This project is intended for educational, research, and experimental AI development purposes.
