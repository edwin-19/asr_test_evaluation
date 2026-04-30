import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import itertools as it

from nemo.collections.asr.models import EncDecCTCModelBPE
from omegaconf import OmegaConf

# Paths
TEST_DIR = os.environ.get("TEST_DIR", "/data/test")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")

# Model weights are bundled alongside this script in the "model/" directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# Target sample rate for Whisper
TARGET_SR = 16000

def pad(audios, audio_lens):
    max_len = max(audio_lens)
    padded_audios = []
    for audio, audio_len in zip(audios, audio_lens):
        if audio_len < max_len:
            pad = (0, max_len - audio_len)
            audio = torch.nn.functional.pad(audio, pad)
        padded_audios.append(audio)
    padded_audios = torch.stack(padded_audios)
    return padded_audios, torch.tensor(audio_lens)

def viterbi(asr_model, predictions):
    texts = []
    for prediction in predictions:
        tokens = [g[0] for g in it.groupby(prediction.detach().cpu().numpy().tolist()) if g[0] != asr_model.decoding.blank_id]
        text = asr_model.decoding.tokenizer.ids_to_text(tokens)
        texts.append(text)
    return texts

def infer(asr_model, audio_array, device):
    audio = torch.tensor(audio_array)
    padded_audios, audio_lens = pad([audio], [audio.shape[0]])
    logits, lengths, greedy_predictions = asr_model.forward(input_signal=padded_audios.to(device), input_signal_length=audio_lens.to(device))
    
    # logits_2d = logits[0]
    preds_text = viterbi(asr_model, greedy_predictions)
    
    return preds_text

def _load_audio(path: str) -> np.ndarray:
    """Load audio file and resample to 16 kHz mono float32."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly

        gcd = np.gcd(TARGET_SR, sr)
        audio = resample_poly(audio, TARGET_SR // gcd, sr // gcd)
    return audio


def main() -> None:
    device = torch.device("cpu")

    print(f"Loading model from {MODEL_DIR} ...")
    
    model_path = os.path.join(MODEL_DIR, 'conformer-averaged.ckpt')
    ckpt = torch.load(model_path, weights_only=False)
    state_dict = ckpt
    
    train_config = OmegaConf.load(os.path.join(MODEL_DIR, 'conformer_ctc_bpe.yaml'))
    train_config.model.tokenizer.dir = os.path.join(MODEL_DIR, 'tokenizer')
    train_config.model.train_ds.manifest_filepath = None
    train_config.model.validation_ds.manifest_filepath = None
    asr_model = EncDecCTCModelBPE(cfg=train_config.model)
    asr_model.load_state_dict(state_dict, strict=True)
    asr_model.to(device)
    asr_model.eval()
    print("Model loaded (CPU).")

    wav_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".wav"))
    print(f"Found {len(wav_files)} audio files.")

    predictions = []

    for i, wav_file in enumerate(wav_files):
        file_id = os.path.splitext(wav_file)[0]
        wav_path = os.path.join(TEST_DIR, wav_file)

        start = time.perf_counter()

        audio = _load_audio(wav_path)
        with torch.no_grad():
            pred_text = infer(asr_model, audio, device)[0]            

        elapsed = time.perf_counter() - start

        predictions.append({
            "file_id": file_id,
            "text": pred_text,
            "inference_time": round(elapsed, 4),
        })

        if (i + 1) % 100 == 0 or i == len(wav_files) - 1:
            print(f"  {i + 1}/{len(wav_files)} done")

    # Write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    total_time = sum(p["inference_time"] for p in predictions)
    print(f"Wrote {len(predictions)} predictions to {out_path}")
    print(f"Total inference time: {total_time:.1f}s")


if __name__ == "__main__":
    main()