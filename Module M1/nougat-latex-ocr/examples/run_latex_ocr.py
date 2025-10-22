# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import argparse

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex.util import process_raw_latex_code
from nougat_latex import NougatLaTexProcessor
import argparse
import os
from nltk import edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def parse_option():
    parser = argparse.ArgumentParser(prog="nougat batch inference with metrics", description="OCR evaluator")
    parser.add_argument("--pretrained_model_name_or_path", default="Norm/nougat-latex-base")
    parser.add_argument("--img_folder", required=True, help="Path to folder containing image files")
    parser.add_argument("--gt_file", required=True, help="Path to .txt file with ground truth LaTeX (line-by-line)")
    parser.add_argument("--device", default="gpu")
    return parser.parse_args()

def compute_metrics(preds, gts):
    assert len(preds) == len(gts)
    edit_dists = []
    bleus = []
    smoothie = SmoothingFunction().method4
    for pred, gt in zip(preds, gts):
        ed = edit_distance(pred, gt) / max(len(pred), len(gt))
        edit_dists.append(ed)
        bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)
        bleus.append(bleu)
    return {
        "final_edit_dist": sum(edit_dists) / len(edit_dists),
        "bleu": sum(bleus) / len(bleus)
    }

def run_batch_nougat():
    args = parse_option()
    device = torch.device("cuda:0") if args.device == "gpu" else torch.device("cpu")

    # Load tokenizer & processor
    tokenizer = AutoTokenizer.from_pretrained(args.config_source)
    image_processor = NougatBucketProcessor.from_pretrained(args.config_source)

    # Initialize encoder and decoder
    encoder = HybridResNetSwinEncoder(pretrained_swin_name=args.config_source)
    decoder_config = MBartConfig.from_pretrained(args.config_source)
    decoder = MBartForCausalLM(config=decoder_config)

    # Build the full model config
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    config.encoder.image_size = [696, 696]  # Match training
    config.decoder.max_length = 2048
    config.decoder.vocab_size = decoder.config.vocab_size

    # Initialize full hybrid model
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    # Load weights
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Load image file paths
    image_files = sorted([
        os.path.join(args.img_folder, f)
        for f in os.listdir(args.img_folder)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    # Load ground truth lines
    with open(args.gt_file, "r", encoding="utf-8") as f:
        gt_lines = [line.strip() for line in f.readlines()]

    assert len(image_files) == len(gt_lines), "Mismatch: #images != #ground-truth lines"

    predictions = []

    for img_path in image_files:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = processor(image, return_tensors="pt").pixel_values
        task_prompt = tokenizer.bos_token
        decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        with torch.no_grad():
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_length,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        seq = tokenizer.batch_decode(outputs.sequences)[0]
        seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
        seq = process_raw_latex_code(seq)
        predictions.append(seq)

    # Evaluate predictions
    metrics = compute_metrics(predictions, gt_lines)

    print("\n=== Evaluation Summary ===")
    print(f"Total Samples: {len(predictions)}")
    print(f"Final Edit Distance (avg): {metrics['final_edit_dist']:.4f}")
    print(f"BLEU Score (avg): {metrics['bleu']:.4f}")

if __name__ == '__main__':
    run_batch_nougat()
