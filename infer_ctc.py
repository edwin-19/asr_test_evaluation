import typer
from nemo.collections.asr.models import EncDecCTCModelBPE
import torch
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import re
import soundfile as sf
import itertools as it
import werpy
import os

try:
    from pyctcdecode import build_ctcdecoder
except:
    print("Error importing ctc decoder")


app = typer.Typer()

def read_json(json_path):
    with open(json_path, 'r') as jsonf:
        return [json.loads(f) for f in jsonf.readlines()]
    
def write_file(text_path, data):
    with open(text_path, 'w') as txtf:
        txtf.writelines(data)

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

def infer(asr_model, decoder, audio_array, device, decoder_type='kenlm'):
    audio = torch.tensor(audio_array)
    padded_audios, audio_lens = pad([audio], [audio.shape[0]])
    logits, lengths, greedy_predictions = asr_model.forward(input_signal=padded_audios.to(device), input_signal_length=audio_lens.to(device))
    
    if decoder_type == 'kenlm':
        logits_2d = logits[0].detach().cpu().numpy()
        text = decoder.decode(logits_2d)
        clean_text = re.sub(r'<s>\s*|\s*</s>', '', text).strip()
        preds_text = [clean_text]
    else:
        preds_text = viterbi(asr_model, greedy_predictions)
    
    return preds_text

@app.command()
def main(
    data_path:str=typer.Option("./data/my_data_train/validation.json"),
    model_path:str=typer.Option("nemo_experiments/conformer-malay-averaged.ckpt"),
    config:str=typer.Option("configs/conformer_ctc_bpe.yaml"),
    device:str=typer.Option("cuda"),
    kenlm_path:str=typer.Option("lm/model.arpa"),
    decoder_type:str=typer.Option("greedy"),
    output_dir:str=typer.Option("output/conformer_ctc_greedy")
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    train_config = OmegaConf.load(config)
    train_config.model.tokenizer.dir = 'tokenizer/tokenizer_spe_bpe_v64_bos_eos'
    asr_model = EncDecCTCModelBPE(cfg=train_config.model)
    asr_model.load_state_dict(state_dict, strict=True)
    asr_model.to(device)
    asr_model.eval()
    
    vocab = asr_model.tokenizer.vocab
    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=kenlm_path,
        alpha=0.6,  # Weight for KenLM (adjust based on performance)
        beta=1.5,   # Weight for word count penalty
    )
    
    manifest = read_json(data_path)
    all_pred = []
    all_gt = []
    for meta in tqdm(manifest):
        new_audio_path = meta['audio_filepath']
        audio, sr = sf.read(new_audio_path)
        with torch.no_grad():
            pred_text = infer(asr_model, decoder, audio, device, decoder_type=decoder_type)[0]
        gt_text = meta['text']
        all_gt.append(gt_text)
        all_pred.append(pred_text)
            
    total_wer = werpy.wer(all_gt, all_pred)
    total_wer = round(total_wer * 100, 4)
    print(f'Total WER: {total_wer}')

    wers = werpy.wers(all_gt, all_pred)
    summary = werpy.summary(all_gt, all_pred)
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    
    all_data = []
    for gt, pred, wer in zip(all_gt, all_pred, wers):
        all_data.append(f"GT: {gt}\nPred: {pred}\nWER: {wer}\n\n")
        
    all_data.append(f"Total WER: {total_wer}")
    write_file(os.path.join(output_dir, 'results.txt'), all_data)

if __name__ == "__main__":
    app()