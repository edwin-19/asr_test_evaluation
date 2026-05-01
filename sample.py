import typer
import json
import os
import random
import shutil
from utils import write_json

app = typer.Typer()

def read_json(json_path):
    with open(json_path, 'r') as jsonf:
        return [json.loads(f) for f in jsonf.readlines()]

def write_file(text_path, data):
    with open(text_path, 'w') as txtf:
        txtf.writelines(data)

@app.command()
def main(
    data_path:str=typer.Option("./data/my_data_train/train.json"),
    write_path:str=typer.Option("data/magic_data/train_lm.txt")
):
    metadata = read_json(data_path)
    all_text = []
    for meta in metadata:
        all_text.append(meta['text'] + '\n')
        
    write_file(write_path, all_text)
    
@app.command()
def sample_data(
    data_path:str=typer.Option("./data/my_data_train/validation.json"),
    write_path:str=typer.Option("sample/")
):
    wav_dir = os.path.join(write_path, 'wavs')
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    
    metadata = read_json(data_path)
    sample_meta = random.sample(metadata, 10)
    for sam in sample_meta:
        shutil.copy(sam['audio_filepath'], wav_dir)
        sam['audio_filepath'] = os.path.join(wav_dir, os.path.basename(sam['audio_filepath']))
        
    write_json(os.path.join(write_path, 'meta.json'), sample_meta)
        

if __name__ == "__main__":
    app()