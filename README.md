# Revolab ASR Train Malay
- Code base for ASR Finetune and Inference using nemo and conformer

# Prerequiest
```
pip install -r requirements.txt
```

# Steps 
1. Build Dataset
```
python process_dataset.py
```

2. Build Tokenizer
```bash
python process_asr_tokenizer.py --manifest data/magic_data/train_sample.json --data_root ./tokenizer/ --vocab_size=64  --tokenizer="spe" --spe_bos --spe_eos
```

3. Train Acoustic Model
```bash
python train_ctc.py
```

3.5 Build KenLM, Optional (But you do need to install KenLM on your local)
```bash
bash build_lm.sh
```

4. Run inference, you can check under output for more detailed break down
```bash
python infer_ctc.py --model-path 
```

# Model Summary Table

| Model Name | Metric | Value (WER) | Download Link |
| :--- | :--- | :--- | :--- |
| **Conformer CTC Model** | Word Error Rate (WER) | 7.9 | [Google Drive Link](https://drive.google.com/file/d/1LDp1O_Yq38f636b1hsm5hix-YctPEHm2/view?usp=drive_link) |
| **Language Model (LM)** | Word Error Rate (WER) | 7.1| [Link TBD] |
| **Conformer RNN-T Model** | Word Error Rate (WER) | [Pending Evaluation] | [Link TBD] |


