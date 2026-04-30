import argparse
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import torch
import os
import onnx
# from onnxruntime.quantization import QuantType, quantize_dynamic
# from utils import optimize_model

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint", required=False, default='./nemo_experiments/Conformer-CTC-BPE-Malay/2026-04-29_13-45-54/checkpoints/Conformer-CTC-BPE-Malay--val_wer=0.1644-epoch=88.ckpt', type=str, help="The path of the checkpoint file of the ASR model")
    args.add_argument("--cfg", required=False, type=str, default='configs/conformer_ctc_bpe.yaml', help="The path of the cfg file of the ASR model")
    args.add_argument("--tokenizer", required=False, type=str, default='tokenizer/tokenizer_spe_bpe_v64_bos_eos', help="The path of the cfg file of the ASR model")
    args.add_argument('--outdir', default='outpath')
    args.add_argument('--is_quant', action='store_true')
    args.add_argument("--autocast", action="store_true", help="Use autocast when exporting")
    args.add_argument("--runtime-check", action="store_true", help="Runtime check of exported net result")
    args.add_argument("--check-tolerance", type=float, default=0.01, help="tolerance for verification")
    args.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")
    args.add_argument("--max-batch", type=int, default=None, help="Max batch size for model export")
    args.add_argument("--max-dim", type=int, default=None, help="Max dimension(s) for model export")
    args.add_argument("--onnx-opset", type=int, default=None, help="ONNX opset for model export")
    args.add_argument(
        "--cache_support", action="store_true", help="enables caching inputs for the models support it."
    )
    args.add_argument("--device", default="cuda", help="Device to export for")

    return args.parse_args()

def load_model(model_path, config_path, tokenizer_path, device='cpu'):
    cfg = OmegaConf.load(config_path)
    cfg.model.tokenizer.dir = tokenizer_path
    model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg.model)
    model.to(device)
    #model.load_state_dict(torch.load(args.checkpoint, map_location=self.device)['state_dict'], strict=True)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    return model

if __name__ == "__main__":
    args = get_args()

    model = load_model(args.checkpoint, args.cfg, args.tokenizer)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(os.path.join(args.outdir, "tokens.txt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(model.decoder.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    with torch.no_grad():
        encoder_filename = "{}/model_encoder_ctc.onnx".format(args.outdir)  # model_encoder_ctc.onnx
        model.encoder.export(encoder_filename)
       
        full_filename = "{}/model_ctc.onnx".format(args.outdir)
        model.export(full_filename)

        decoder_filename = "{}/model_decoder_ctc.onnx".format(args.outdir)
        model.decoder.export(decoder_filename)

        normalize_type = model.cfg.preprocessor.normalize
        if normalize_type == "NA":
            normalize_type = ""

        model_name = 'conformer_ctc_large'
        meta_data = {
            "vocab_size": model.tokenizer.vocab_size,
            "normalize_type": normalize_type,
            "subsampling_factor": 4,
            "model_type": "EncDecCTCModelBPE",
            "version": "1",
            "model_author": "NeMo",
            "url": f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/{model_name}",
            "comment": "Only the CTC branch is exported",
            "doc": "",
        }

        print("preprocessor", model.cfg.preprocessor)
        print(meta_data)
        
        # optimize_model(
        #     encoder_filename, 'outpath/model_encoder_optimised.onnx',
        #     model.cfg.encoder.n_heads, model.cfg.encoder.d_model, model_type='conformer'
        # )

    # if args.is_quant:
    #     print("Quantizing Model to int8")
        
    #     filename = 'outpath/model_ctc.onnx'
        
    #     op_types_to_quantize = ["Attention", "CrossAttention", "RelPosAttention", "MatMul",]
        
    #     quantize_dynamic(
    #         model_input=filename,
    #         model_output=filename.replace('.onnx', '.int8.onnx'),
    #         per_channel=True,
    #         weight_type=QuantType.QUInt8,
    #         op_types_to_quantize=op_types_to_quantize
    #     )
