from pathlib import Path
import copy
import re

from tqdm import tqdm
import editdistance
import json

from nemo.utils import logging
import regex as re_
from omegaconf import OmegaConf

def write_file(file_path, data):
    with open(file_path, 'w') as text_file:
        text_file.writelines(data)

def read_file(file_path):
    with open(file_path, 'r') as text_file:
        return [t.replace('\n', '') for t in text_file.readlines()]

def write_json(file_path, data):
    data = [json.dumps(d, ensure_ascii=False) + '\n' for d in data]
    with open(file_path, 'w') as manifest_file:
        manifest_file.writelines(data)

def read_json(json_path):
    loaded_data = []
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json_file.readlines()
        for index, d in enumerate(data):
            try:
                d = fix_json(d)
                loaded_data.append(json.loads(d))
            except Exception as err:
                print(err, '\n',d, index)
    return loaded_data

def fix_json(json_str):
    # Find the index of "duration" key
    index = json_str.find('"duration":')
    modified_json_str = json_str
    if index != -1:
        # Locate the start and end of the value for "duration"
        value_start = index + len('"duration":') + 1
        value_end = json_str.find(',', value_start)  # Assuming value is followed by a comma

        # Remove double quotes from the value
        duration_value = json_str[value_start:value_end]
        duration_value = duration_value.strip('"')  # Remove surrounding double quotes

        # Construct the modified JSON string
        modified_json_str = json_str[:value_start] + duration_value + json_str[value_end:]
    
    return modified_json_str

def get_stats(manifest):
    durations = [mani['duration'] for mani in manifest]
    print('Total Lines: {}'.format(len(manifest)))    
    print('Total Hours: {}\nMin: {}\nMax: {}\nAvg: {}\n'.format(
            sum(durations) / 3600,
            min(durations), max(durations), sum(durations) / len(durations)
        )
    )

def compute_wer(hyps, refs):
    error_total = 0
    length_total = 0
    for hyp, ref in zip(hyps, refs):
        hyp_words = hyp.split()
        ref_words = ref.split()
        error = editdistance.eval(hyp_words, ref_words)
        error_total += error
        length_total += len(ref_words)
    wer = error_total * 100.0 / length_total
    return wer

def compute_cer(hyps, refs):
    error_total = 0
    length_total = 0
    for hyp, ref in zip(hyps, refs):
        error_total += editdistance.eval(list(hyp), list(ref))
        length_total += len(ref)
    cer = error_total * 100.0 / length_total
    return cer

def write_file(txt_path, data):
    with open(txt_path, 'w') as txt_file:
        txt_file.writelines(data)

def get_user_input_value(s):
  if not s:
      return None
  s = s.strip()
  if s == '':
      return None
  if (s.startswith('[') and s.endswith(']') or
      s.startswith('{') and s.endswith('}') ):
      try:
          x = eval(s)
      except Exception as e:
          logging.error(f"## failed to evaluate '{s}' ...")
          return None
      return x
  try:
      x = int(s)
      return x
  except ValueError:
      try:
          x = float(s)
          return x
      except ValueError:
          x = str(s)
          if x.lower() == 'true':
              return True
          if x.lower() in ['false']:
              return False    
          return x 

def parse_command_config(argv, original_config=None):
    str_argv = ' '.join(argv)
    argv_list = str_argv.split('--')
    argv_list = [x.strip() for x in argv_list if x.strip() !='']
    argv_list = [x.replace(' ', '=')  for x in argv_list]
    cmd_conf = OmegaConf.create({})
    for argv in argv_list:
        item = argv.split('=')
        assert len(item) == 2, f"illegal arguments '{argv}' ..."
        value = get_user_input_value(item[1])
        cmd_conf.update({item[0]: value})

    cfg_file = cmd_conf.get('cfg', None)
    conf_conf = None
    if cfg_file:
      cmd_conf.pop('cfg')
      try:
        conf_conf = OmegaConf.load(cfg_file)
      except Exception as  e:
        logging.error(f"## failed to load cfg_file '{cfg_file}' ...")
        return None
    else: conf_conf = original_config
    if not conf_conf:
      return None
    optim = OmegaConf.select(conf_conf, 'model.optim')
    for key, val in cmd_conf.items():
        if OmegaConf.select(conf_conf, key) is None:
            logging.warning(f"WARNING: no key '{key}', insertion proceeds ...")
        OmegaConf.update(conf_conf, key, val, merge=True)
    return conf_conf


def read_manifest(manifest_path):
    with open(manifest_path, 'r') as manifest:
        return [json.loads(mani) for mani in manifest.readlines()]
    

def clean_punctuation_spacing(text):
    # \s+ looks for one or more whitespace characters
    # (?=[.,!?]) is a lookahead that ensures they are followed by punctuation
    # This replaces "word ." with "word."
    cleaned_text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # Optional: Ensure there is exactly one space AFTER punctuation if it's not the end of the string
    cleaned_text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', cleaned_text)
    cleaned_text = cleaned_text.replace('-', ' ')
    
    return cleaned_text.strip()