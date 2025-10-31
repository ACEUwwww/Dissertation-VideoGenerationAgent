# Sora -> LLM -> TTS -> Wav2Lip Orchestrator

This repository contains `agent.py`, a lightweight orchestrator that demonstrates a pipeline:

- Generate or accept a silent video (Sora via OpenAI Video.create or a local file)
- Generate narration text with OpenAI ChatCompletion (GPT-4)
- Synthesize speech with edge-tts (preferred) or gTTS (fallback)
- Lip-sync audio to the silent video using Wav2Lip (if installed) or fallback to muxing with moviepy

Requirements
 - Python 3.8+
 - Install dependencies:

```powershell
pip install -r requirements.txt
```

Using the script

Generate everything (Sora video requires OpenAI account with Video API access):

```powershell
python agent.py
```

Wav2Lip
To use Wave2lip, need to first train the model or get a weight online.
Put the .pth file to Wave2lip/checkpoints

Security
- Do NOT hardcode API keys. Use environment variables. The original `api.py` contained an exposed key; it has been removed.
