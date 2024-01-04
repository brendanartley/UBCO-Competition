from ubco_stage2.train import train

import argparse
from types import SimpleNamespace
import json, os

# Load environment variables (stored in config.json file)
with open('./config.json') as f:
    data = json.load(f)
DATA_DIR = data["DATA_DIR"]

# default configuration parameters
config = SimpleNamespace(
    project = "UBCO",
    data_dir = DATA_DIR,
    hf_cache = os.path.join(DATA_DIR, "HF_CACHE/"),
    torch_cache = os.path.join(DATA_DIR, "TORCH_CACHE/"),
    model_name = "tf_efficientnetv2_b2.in1k",
    pretrained_weights = "",
    num_classes = 5,
    tile_size = 1280,
    num_tiles = 8,
    top_n_imgs = 24,
    min_n_imgs = 12,
    batch_size = 7,
    epochs = 18,
    lr = 1e-4,
    label_smoothing = 0.0,
    scheduler = "CosineAnnealingLR",
    save_weights = False,
    val_tta = 5,
    val_fold = 0,
    weight_decay = 1e-4,
    ham_pct = 0.10,
    # -- Other --
    accelerator = "gpu",
    fast_dev_run = False,
    overfit_batches = 0,
    strategy = "auto",
    precision = "16-mixed", # -true, -mixed suffix
    accumulate_grad_batches = 2,
    num_workers = 4,
    seed = 0,
    verbose = 2,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fast_dev_run', action='store_true', help='Debug run.')
    parser.add_argument("--model_name", type=str, default=config.model_name)
    parser.add_argument("--pretrained_weights", type=str, default=config.pretrained_weights)
    parser.add_argument("--overfit_batches", type=int, default=config.overfit_batches, help="Test model can overfit (sanity check).")
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--accumulate_grad_batches", type=int, default=config.accumulate_grad_batches)
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--valid_only', action='store_true')
    parser.add_argument("--val_tta", type=int, default=config.val_tta)
    parser.add_argument("--val_fold", type=int, default=config.val_fold)
    parser.add_argument("--top_n_imgs", type=int, default=config.top_n_imgs)
    parser.add_argument("--min_n_imgs", type=int, default=config.min_n_imgs)
    parser.add_argument("--tile_size", type=int, default=config.tile_size)
    parser.add_argument("--num_tiles", type=int, default=config.num_tiles)
    parser.add_argument("--label_smoothing", type=float, default=config.label_smoothing)
    parser.add_argument("--precision", type=str, default=config.precision)
    parser.add_argument('--train_all_data', action='store_true')
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--ham_pct", type=float, default=config.ham_pct, help="High Attention Masking %")
    parser.add_argument("--num_workers", type=int, default=config.num_workers)
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config


def main(config):
    train(config)
    return

if __name__ == "__main__":
    config = parse_args()
    main(config)