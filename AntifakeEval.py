"""
 - A100 0.2s / frame

Usage:
conda activate antifake310
python3 "AntifakeEval.py" \
    --data_root "${data_root}" \
    --pred_csv "${result_dir}/predictions.csv"
"""
import argparse
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from DataUtils import IMG_EXTS, index_dataframe, REQUIRED_COLS, standardise_predictions
from lavis.models import load_model_and_preprocess


class FakePartV2Dataset(Dataset):
    def __init__(self, data_root, vis_proc=None):
        self.data_root = Path(data_root)
        self.df = index_dataframe(self.data_root, file_exts=IMG_EXTS)
        # keep only the columns we need to fill REQUIRED_COLS later
        keep = ["task", "method", "subset", "label", "mode", "rel_path", "abs_path"]
        self.df = self.df[keep].reset_index(drop=True)
        self.vis_proc = vis_proc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        if self.vis_proc is not None:
            img = self.vis_proc(img)
        # minimal metadata needed to populate REQUIRED_COLS
        meta = {
            "sample_id": row["rel_path"],  # unique id; relative path is convenient
            "task": row["task"],
            "method": row["method"],
            "subset": row["subset"],
            "label": int(row["label"]),  # 0=real, 1=fake (as defined in DataUtils)
            "mode": row["mode"],  # 'frame' or 'video' (here: frames)
        }
        return img, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Dataset root to index.")
    p.add_argument("--pred_csv", type=str, required=True, help="Output CSV path.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=os.cpu_count())
    p.add_argument("--model_name", type=str, default="blip2_vicuna_instruct_textinv")
    p.add_argument("--model_type", type=str, default="vicuna7b")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # Load model + preprocessors
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.model_name, model_type=args.model_type, is_eval=True, device=device
    )
    vis_eval = vis_processors.get("eval", None)

    # Build dataset/dataloader
    dataset = FakePartV2Dataset(args.data_root, vis_proc=vis_eval)
    if len(dataset) == 0:
        # Write an empty CSV with REQUIRED_COLS header, then exit gracefully.
        empty = pd.DataFrame(columns=list(REQUIRED_COLS))
        Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(args.pred_csv, index=False)
        print(f"[info] No samples found under {args.data_root}. Wrote empty file: {args.pred_csv}")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Simple binary Q&A: "Is this photo fake?" => yes=1 (fake), no=0 (real)
    question = "Is this photo fake?"
    candidates = ["yes", "no"]  # we will map yes->1 (fake), no->0 (real)

    rows = []
    model_id = args.model_name

    with torch.no_grad():
        for images, metas in tqdm(loader, desc="Evaluating"):
            images = images.to(device)

            # Use LAVIS VQA API (answers list). Keeps it robust and simple.
            questions = [question] * images.shape[0]
            # samples = {"image": images, "text_input": questions}
            # answers = model.predict_answers(samples=samples, inference_method="generate", answer_list=candidates)

            # Map to hard pred in {0,1}; score is unavailable -> -1.0
            # preds = [1 if a == "yes" else 0 for a in answers]
            # scores = [-1.0] * len(preds)

            samples = {"image": images, "prompt": questions}
            candidates = ["yes", "no"]
            ans = model.predict_class(samples=samples, candidates=candidates)

            preds = []
            for a in ans:
                # if a is already an int index:
                if isinstance(a, (int, np.integer)):
                    idx = int(a)
                # if a is a 1D score/one-hot vector:
                elif hasattr(a, "__len__"):
                    idx = int(torch.argmax(a).item())
                else:
                    raise ValueError(f"Unexpected predict_class output: {type(a)}")
                preds.append(1 if candidates[idx] == "yes" else 0)  # yes->fake(1), no->real(0)
            scores = [-1.0] * len(preds)

            # Pack rows in REQUIRED_COLS order
            for i in range(len(preds)):
                rows.append({
                    "sample_id": metas["sample_id"][i],
                    "task": metas["task"][i],
                    "method": metas["method"][i],
                    "subset": metas["subset"][i],
                    "label": int(metas["label"][i]),
                    "model": model_id,
                    "mode": metas["mode"][i],
                    "score": float(scores[i]),  # -1.0 when unavailable
                    "pred": int(preds[i]),
                })

    # Standardise + write
    df_out = standardise_predictions(rows)
    Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.pred_csv, index=False)
    print(f"[ok] Wrote predictions to: {args.pred_csv}")
    print(f"[shape] {df_out.shape[0]} rows, columns = {list(df_out.columns)}")


if __name__ == "__main__":
    main()
