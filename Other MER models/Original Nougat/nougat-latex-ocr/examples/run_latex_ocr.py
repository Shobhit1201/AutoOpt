# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import sys
sys.path.append('/content/drive/MyDrive/OriginalNougat/nougat-latex-ocr')

# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import argparse

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex.util import process_raw_latex_code
from nougat_latex import NougatLaTexProcessor


def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="Norm/nougat-latex-base")  # This must be a directory or repo ID
    parser.add_argument("--img_path", help="Path to LaTeX image segment", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--checkpoint_path", help="Path to .pth file containing model weights", required=False)
    return parser.parse_args()

def run_nougat_latex():
    args = parse_option()
    device = torch.device("cuda:0") if args.device == "gpu" else torch.device("cpu")

    # Load model config (not weights) from directory or repo
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model_name_or_path)

    # Load checkpoint weights if provided
    if args.checkpoint_path:
        print(f"Loading weights from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Load tokenizer & processor
    tokenizer = NougatTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
    latex_processor = NougatLaTexProcessor.from_pretrained(args.pretrained_model_name_or_path)

    # Load and process image
    image = Image.open(args.img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = latex_processor(image, return_tensors="pt").pixel_values

    # Generate
    decoder_input_ids = tokenizer(tokenizer.bos_token, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.config.decoder.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    sequence = tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
    sequence = process_raw_latex_code(sequence)
    print(sequence)
    return sequence

if __name__ == "__main__":
    run_nougat_latex()