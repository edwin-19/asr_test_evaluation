import typer
import json
from sklearn.model_selection import train_test_split
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
import os
import shutil

app = typer.Typer()

def write_jsonl(data, data_path):
    data = [json.dumps(d, ensure_ascii=True) + '\n' for d in data]
    with open(data_path, 'w') as jsonf:
        jsonf.writelines(data)

def read_file(text_path):
    with open(text_path, 'r') as txtf:
        return txtf.readlines()

@app.command()
def main(
    data_path:str=typer.Option("data/magic_data/UTTRANSINFO.txt"),
    audio_folder:str=typer.Option("data/magic_data/flac"),
    singlsh_dataset:str=typer.Option('data/singlish-speaker2050'),
    output_path:str=typer.Option("data/my_data_train"),
    sample_size:int=typer.Option(30000)
):
    metadata = read_file(data_path)
    # targets = {'.': 0, ',': 0, '?': 0}
    targets = {'.', ',', '?'}
    mandatory_samples = []
    filler_samples = []
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    dataset = load_dataset(singlsh_dataset)
    subset = dataset['train'].shuffle(seed=42).select(range(len(dataset['train'])))
    
    all_data = []
    wav_path = os.path.join(output_path, 'wav')
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
        
    for index, data in tqdm(enumerate(subset)):
        duration = len(data['audio']['array']) / data['audio']['sampling_rate']
        audio_path = os.path.join(wav_path, data['audio']['path'])
        sf.write(audio_path, data['audio']['array'], data['audio']['sampling_rate'])
        all_data.append({
            'audio_filepath': audio_path,
            "text": data['transcription'],
            'duration': duration
        })
    
    for i, line in tqdm(enumerate(metadata[1:])):
        tokens = line.strip().split('\t')
        name, text = tokens[1], tokens[4]
        folder = name.split('_')[0]
        
        audio_file = f'{audio_folder}/{folder}/{name}'.replace('.wav', '.flac')
        if any(mark in text for mark in targets):
            y, sr = sf.read(audio_file)
            duration = len(y) / sr
            
            shutil.copy(audio_file, wav_path)
            new_wav_path = os.path.join(wav_path, os.path.basename(audio_file))
            mandatory_samples.append({
                'audio_filepath': new_wav_path,
                'duration': round(duration, 2),
                'text': text.lower()
            })
            continue
        
        if len(filler_samples) < sample_size:
            try:
                y, sr = sf.read(audio_file)
                duration = len(y) / sr
                new_wav_path = os.path.join(wav_path, os.path.basename(audio_file))
                shutil.copy(audio_file, wav_path)
                filler_samples.append({
                    'audio_filepath': new_wav_path,
                    'duration': round(duration, 2),
                    'text': text.lower()
                })
            except Exception as err:
                print(err)
                continue

    total_samples = mandatory_samples + filler_samples + all_data
    train_samples, test_samples = train_test_split(total_samples, test_size=0.2, random_state=42)
    write_jsonl(train_samples, os.path.join(output_path, 'train.json'))
    write_jsonl(test_samples, os.path.join(output_path, 'validation.json'))

if __name__ == "__main__":
    app()