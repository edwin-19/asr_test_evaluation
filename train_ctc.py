import argparse
from utils import parse_command_config
# import pytorch_lightning as pl
from lightning.pytorch import Trainer, LightningModule
import torch

from omegaconf import OmegaConf
from nemo.utils import logging
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecCTCModelBPE
from nemo.utils.exp_manager import exp_manager

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--conf', type=str, default='configs/conformer_ctc_bpe.yaml', required=False)
    parse.add_argument('--monitor', type=str, default='val_wer', help='')
    parse.add_argument('--save_top_k', type=int, default=10, help='')
    parse.add_argument('--checkpoint_path', type=str, default='checkpoints', help='')
    parse.add_argument('--devices', type=int, default=4, help='')
    parse.add_argument('--accelerator', type=str, default='gpu', help='')
    parse.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true')
    parse.add_argument('--max_epochs', type=int, default=200, help='')
    parse.add_argument('--num_nodes', type=int, default=1, help='')
    parse.add_argument('--log_every_n_steps', type=int, default=2000, help='')
    parse.add_argument('--check_val_every_n_epoch', default=None, help='')
    parse.add_argument('--accumulate_grad_batches', type=int, default=8, help='')
    #parse.add_argument('--resume_from_checkpoint', type=str, default=None, help='')
    parse.add_argument('--new_tokenizer_dir', type=str, default='tokenizer', help='')
    parse.add_argument('--new_tokenizer_type', type=str, default='bpe', help='')
    parse.add_argument('--freeze_encoder', action='store_true', help='')
    parse.add_argument('--val_check_interval', type=int, default=2000, help='')
    args, options = parse.parse_known_args()

    return args, options

def main():
    args, options = get_args()
    logging.info(f"## config file: '{args.conf}' ...")
    train_config = OmegaConf.load(args.conf)
    train_config = parse_command_config(options, train_config)

    logging.info(f"config={OmegaConf.to_yaml(train_config)}")

    trainer = Trainer(**train_config.trainer)
    exp_manager(trainer, train_config.get("exp_manager", None))
    asr_model = EncDecCTCModelBPE(cfg=train_config.model, trainer=trainer)

    # Load model
    logging.info("Using Checkpoint: {}".format(train_config.get("encoder_ckpt")))
    ckpt = torch.load(train_config.get("encoder_ckpt", "models/mtl_54k.ckpt"), map_location='cuda', weights_only=False)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    keys_to_delete = [k for k in state_dict.keys() if "decoder.decoder_layers" in k]
    for k in keys_to_delete:
        del state_dict[k]
        logging.info(f"Removed {k} from checkpoint to avoid size mismatch.")

    asr_model.load_state_dict(state_dict, strict=False)

    # Train model
    trainer.fit(asr_model)

if __name__ == "__main__":
    main()