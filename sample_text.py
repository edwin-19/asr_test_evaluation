import typer
import json

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

if __name__ == "__main__":
    app()