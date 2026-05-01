# Revolab ASR Train Malay
- Code base for ASR Finetune and Inference using nemo and conformer

# Prerequisite
```
pip install -r requirements.txt
```

- Install KENLM (Optional)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev

# Clone the official repository
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Create a build directory
mkdir -p build
cd build

# Run CMake and Compile
cmake ..
make -j$(nproc)

# (Optional) Install binaries to your system path
sudo make install
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

3.8 Average Checkpoint, average out the top 5 checkpoints for better scoring
```bash
python average_model_checkpoints.py
```

4. Run inference, you can check under output for more detailed break down
```bash
python infer_ctc.py --model-path ./model_path/conformer_ctc_averged.ckpt
```

4.5 Run Inference on the RNNT model  (this can be adjusted with beam size, defaults to 2)
```bash
python infer_rnnt.py --beam-size 1 --use-beam
```

# Model Summary Table

| Model Name | Metric | Value (WER) | Download Link |
| :--- | :--- | :--- | :--- |
| **Conformer CTC Model** | Word Error Rate (WER) | 7.22  | [Google Drive Link](https://drive.google.com/file/d/1LDp1O_Yq38f636b1hsm5hix-YctPEHm2/view?usp=drive_link) |
| **CTC + Language Model (LM)** | Word Error Rate (WER) | 6.16 | [Google Drive Link](https://drive.google.com/file/d/1LDp1O_Yq38f636b1hsm5hix-YctPEHm2/view?usp=drive_link) |
| **Conformer RNN-T Model Beam Size 1** | Word Error Rate (WER) | 6.76 | [Google Drive Link](https://drive.google.com/file/d/1As92LO_U-9Q4mZQqIACejbS5qttS0wgl/view?usp=sharing) |
| **Conformer RNN-T Model Beam Size 2** | Word Error Rate (WER) | 6.70 | [Google Drive Link](https://drive.google.com/file/d/1As92LO_U-9Q4mZQqIACejbS5qttS0wgl/view?usp=sharing) |


# Optional Adding in submission style for kaggle
```bash
cd submission

cp ../models/conformer-averaged.ckpt ./model/conformer-averaged.ckpt

# add in any extra packages on the wheel file (in this case nemo toolkit)
pip wheels nemo_toolkit[asr] -w wheels/

# Optional zip -r submission_2.zip inference.py model wheels
python inference.py
```