import sys
sys.path.append('/content/drive/MyDrive/Nougat_scratch_temp1/nougat-latex-ocr')


import argparse
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, MBartForCausalLM
from nougat_latex.util import process_raw_latex_code
from nougat_latex import NougatLaTexProcessor
from experiment.hybrid_encoder_swin import HybridResNetSwinEncoder
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jiwer

def parse_option():
    parser = argparse.ArgumentParser(prog="nougat batch inference with metrics", description="OCR evaluator")
    parser.add_argument("--img_folder", required=True, help="Path to folder containing images")
    parser.add_argument("--gt_file", required=True, help="Path to .txt file containing ground truth LaTeX")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--config_source", default="Norm/nougat-latex-base", help="Tokenizer and processor source")
    return parser.parse_args()

def tokenize_latex(expr):
    # Character-level and LaTeX-level tokenization
    return re.findall(r'(\\[a-zA-Z]+|[{}_^=+\-*/(),]|[a-zA-Z]+|\d+)', expr)

# --- Tokenizer Transform for CER ---
class TokenizeTransform(jiwer.transforms.AbstractTransform):
    def process_string(self, s: str):
        return tokenize_latex(s)

    def process_list(self, tokens: list[str]):
        return [self.process_string(token) for token in tokens]

# --- CER Calculation ---
def compute_cer(truth_and_output: list[tuple[str, str]]):
    ground_truth, model_output = zip(*truth_and_output)
    return jiwer.cer(
        truth=list(ground_truth),
        hypothesis=list(model_output),
        reference_transform=TokenizeTransform(),
        hypothesis_transform=TokenizeTransform()
    )

def compute_metrics(preds, gts):
    assert len(preds) == len(gts)
    bleus = []
    truth_and_preds = []
    smoothie = SmoothingFunction().method4
    for pred, gt in zip(preds, gts):

        # BLEU: tokenized version
        pred_tokens = tokenize_latex(pred)
        gt_tokens = tokenize_latex(gt)
        bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)

        
        bleus.append(bleu)
        truth_and_preds.append((gt, pred))
        cer = compute_cer(truth_and_preds)

    return {
        "bleu": sum(bleus) / len(bleus),
        "cer": cer
    }

def run_batch_nougat():
    args = parse_option()
    device = torch.device("cuda:0") if args.device == "gpu" else torch.device("cpu")

    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(args.config_source)
    image_processor = NougatLaTexProcessor(size=(768, 1024), do_crop_margin=False)

    # Initialize encoder and decoder
    encoder = HybridResNetSwinEncoder(pretrained_swin_name=args.config_source)
    decoder = MBartForCausalLM.from_pretrained(args.config_source)

    # Build model config
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    config.encoder.image_size = [768, 1024]
    config.decoder.max_length = 1500
    config.decoder.vocab_size = decoder.config.vocab_size

    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load images
    image_files = sorted([
        os.path.join(args.img_folder, f)
        for f in os.listdir(args.img_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Load ground truth
    with open(args.gt_file, "r", encoding="utf-8") as f:
        gt_lines = [line.strip() for line in f.readlines()]

    assert len(image_files) == len(gt_lines), "Mismatch: number of images and ground truth lines"

    predictions = []

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

        decoder_input_ids = tokenizer(tokenizer.bos_token, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=config.decoder.max_length,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
        cleaned = decoded.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
        final_output = process_raw_latex_code(cleaned)
        predictions.append(final_output)

    # Evaluation
    metrics = compute_metrics(predictions, gt_lines)

    # Save predictions
    output_file = "predictions_vs_groundtruth.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (pred, gt) in enumerate(zip(predictions, gt_lines)):
            f.write(f"Sample {i+1}\n")
            f.write(f"Ground Truth : {gt}\n")
            f.write(f"Prediction    : {pred}\n")
            f.write("\n")

    print("\n=== First 20 Predictions vs Ground Truth ===")
    for i in range(min(20, len(predictions))):
        print(f"\nSample {i+1}")
        print("Ground Truth : ", gt_lines[i])
        print("Prediction    : ", predictions[i])

    print("\n=== Evaluation Summary ===")
    print(f"Total Samples: {len(predictions)}")
    print(f"BLEU Score (avg): {metrics['bleu']:.4f}")
    print(f"CER Score (avg): {metrics['cer']:.4f}")
    print(f"Saved detailed results to {output_file}")

if __name__ == "__main__":
    run_batch_nougat()
