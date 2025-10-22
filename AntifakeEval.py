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
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union, Optional
import logging
from typing import List, Tuple, Dict, Any, Optional
import torch

from DataUtils import (
    IMG_EXTS, REQUIRED_COLS, standardise_predictions, FRAMES_ROOT, FRAMES_CSV,
    FakePartsV2DatasetBase, collate_skip_none
)
from lavis.models import load_model_and_preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class FakePartsV2Dataset(FakePartsV2DatasetBase):
    def __init__(self, data_root: Union[str, Path] = FRAMES_ROOT,
                 mode: str = "frame",
                 csv_path: Optional[Union[str, Path]] = FRAMES_CSV,
                 model_name: str = "AntiFakeVLM",
                 transform=None,
                 vis_proc=None,
                 on_corrupt: str = "warn",
                 ):
        super().__init__(data_root=data_root, mode=mode, csv_path=csv_path,
                         model_name=model_name, transform=None, on_corrupt=on_corrupt)
        self.vis_proc = vis_proc

    def __getitem__(self, idx):
        x, y, meta = super().__getitem__(idx)
        # base returns: PIL.Image for frames, VideoCapture for videos
        if self.mode == "frame" and self.vis_proc is not None:
            x = self.vis_proc(x)
        return x, y, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Dataset root to index.")
    p.add_argument('--data_csv', type=str, default=None, help='csv file indexing the dataset')
    p.add_argument("--pred_csv", type=str, required=True, help="Output CSV path.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=304)
    p.add_argument("--num_workers", type=int, default=4)
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
    dataset = FakePartsV2Dataset(args.data_root, vis_proc=vis_eval, csv_path=args.data_csv)
    if len(dataset) == 0:
        raise SystemExit(f"No images found under {args.data_root} with extensions {IMG_EXTS}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=collate_skip_none,
        persistent_workers=(args.num_workers or 0) > 0,
    )

    # Simple binary Q&A: "Is this photo fake?" => yes=1 (fake), no=0 (real)
    question = "Is this photo fake?"
    candidates = ["yes", "no"]  # map yes->1 (fake), no->0 (real)
    model_id = args.model_name

    # Init csv header and save
    Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=REQUIRED_COLS).to_csv(args.pred_csv, index=False)

    batch_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                log.warning("Skipping a batch where all samples were corrupt.")
                continue  # whole batch was corrupt; skip
            images, labels, metas = batch
            images = images.to(device)

            # Use LAVIS VQA API (answers list). Keeps it robust and simple.
            questions = [question] * images.shape[0]
            # samples = {"image": images, "text_input": questions}
            # answers = model.predict_answers(samples=samples, inference_method="generate", answer_list=candidates)

            # Map to hard pred in {0,1}; score is unavailable -> -1.0
            # preds = [1 if a == "yes" else 0 for a in answers]
            # scores = [-1.0] * len(preds)

            samples = {"image": images, "prompt": questions}
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
                preds.append(1 if candidates[idx] == "yes" else 0)

            scores = [-1.0] * len(preds)

            # Pack rows in REQUIRED_COLS order
            for i in range(len(preds)):
                batch_rows.append({
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

            if batch_rows:
                df_batch = standardise_predictions(batch_rows)
                df_batch.to_csv(args.pred_csv, mode="a", header=False, index=False)
                batch_rows.clear()

    # Final report: read back and print shape
    df_out = pd.read_csv(args.pred_csv)
    log.info(f"[ok] Wrote predictions to: {args.pred_csv}")
    log.info(f"[shape] {df_out.shape[0]} rows, columns = {list(df_out.columns)}")


if __name__ == "__main__":
    main()
