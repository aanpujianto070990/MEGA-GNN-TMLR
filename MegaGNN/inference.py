import datetime
import logging
import time
import os
import os.path as osp
from typing import List, Dict, Any, Optional
import yaml
import numpy as np
import torch
from torch_geometric import seed_everything

import MegaGNN  # noqa, register custom modules
from MegaGNN.graphgym.config import (cfg, set_cfg, load_cfg)
from MegaGNN.graphgym.loader import create_loader
from MegaGNN.graphgym.logger import setup_printing
from MegaGNN.graphgym.model_builder import create_model
from MegaGNN.graphgym.checkpoint import load_ckpt, get_ckpt_epochs, get_ckpt_path
from MegaGNN.graphgym.utils.device import auto_select_device
from MegaGNN.graphgym.loss import compute_loss
from MegaGNN.graphgym.utils.comp_budget import params_count
from MegaGNN.utils import custom_set_run_dir
from MegaGNN.logger import create_logger
from MegaGNN.train.custom_train import eval_epoch


# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# HARDCODED CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# =============================================================================

# Path to the model directory containing config.yaml and ckpt/
MODEL_PATH = "/tudelft.net/staff-umbrella/ScalableGraphLearning/cagri/exps/neurips/ETH-Multi-FraudGT"  # CHANGE THIS
# List of seeds to run inference on
SEEDS = [33, 43, 76, 96, 97]  # CHANGE THIS
# GPU device ID to use (-1 for auto selection)
GPU = 1  # CHANGE THIS
# Checkpoint epoch to load (-1 for largest/latest checkpoint number)
EPOCH = -1  # CHANGE THIS
# Dataset split to run inference on ('test' or 'val')
SPLIT = 'test'  # CHANGE THIS
# Enable verbose logging
VERBOSE = True  # CHANGE THIS
# Permute ports
PERMUTE_PORTS = True  # CHANGE THIS


# =============================================================================


def get_inference_args():
    """Get hardcoded inference arguments."""
    class Args:
        def __init__(self):
            self.model_path = MODEL_PATH
            self.seeds = SEEDS
            self.gpu = GPU
            self.epoch = EPOCH
            self.split = SPLIT
            self.verbose = VERBOSE
            self.permute_ports = PERMUTE_PORTS
    
    return Args()

    
def setup_config_from_model_path(model_path: str, seed: int, args) -> None:
    """Setup configuration from model path and seed."""
    # Set base configuration
    set_cfg(cfg)
    # Update run_dir to point to the model path for checkpoint loading
    run_dir = osp.join(model_path, f'{seed}')
    cfg.run_dir = run_dir
    cfg.merge_from_file(osp.join(run_dir, 'config.yaml'))
    cfg.run_dir = run_dir
    cfg.dataset.task_entity = tuple(cfg.dataset.task_entity)
    


def setup_environment(gpu: int) -> None:
    """Setup PyTorch environment and device."""
    torch.set_num_threads(cfg.num_threads)
    
    if gpu == -1:
        auto_select_device(strategy='greedy')
    else:
        logging.info(f'Select GPU {gpu}')
        if cfg.device == 'auto':
            cfg.device = f'cuda:{gpu}'



def run_inference_single_condition(model_path: str, seed: int, args, permute_ports: bool) -> Dict[str, Any]:
    """Run inference for a single seed with specific permute_ports setting."""
    logging.info(f"Running inference for seed {seed} with permute_ports={permute_ports}")
    
    # Setup configuration for this seed
    setup_config_from_model_path(model_path, seed, args)
    seed_everything(seed)
    cfg.dataset.permute_ports = permute_ports
    cfg.dataset.task_entity = "node"
    
    # Create dataset and loaders
    loaders, dataset = create_loader(returnDataset=True)
    loggers = create_logger()



    # Create model
    model = create_model(dataset=dataset)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    
    # Find the latest checkpoint
    ckpt_dir = osp.join(cfg.run_dir, 'ckpt')
    filename = os.listdir(ckpt_dir)[0]   
    epoch = int(filename.split('.')[0])     
    
    # Load the latest checkpoint
    ckpt = torch.load(osp.join(ckpt_dir, filename))
    model.load_state_dict(ckpt['model_state'])
    
    
    # Get test loader
    split_idx = 2 if args.split == 'test' else 1  # test=2, val=1 (train=0)
    if split_idx >= len(loaders):
        raise ValueError(f"Split '{args.split}' not available. Available splits: {len(loaders)}")
    
    
    eval_epoch(loggers[split_idx], loaders[split_idx], model, args.split)
    results = loggers[split_idx].write_epoch(epoch)
    
    return {'seed': seed, 'epoch': epoch, 'permute_ports': permute_ports, 'metrics': results}


def run_inference_single_seed(model_path: str, seed: int, args) -> List[Dict[str, Any]]:
    """Run inference for a single seed with both permute_ports=False and True."""
    results = []
    
    # Run with permute_ports=False
    result_original = run_inference_single_condition(model_path, seed, args, permute_ports=False)
    results.append(result_original)
    
    # Run with permute_ports=True  
    result_true = run_inference_single_condition(model_path, seed, args, permute_ports=True)
    results.append(result_true)
    
    # Print comparison for this seed
    print(f"\nSeed {seed} Results Comparison:")
    print(f"{'Metric':<15} {'No Permute':<12} {'Permuted':<12} {'Difference':<12}")
    print("-" * 55)
    
    for metric in result_original['metrics'].keys():
        val_false = result_original['metrics'][metric]
        val_true = result_true['metrics'][metric]
        diff = val_true - val_false
        print(f"{metric:<15} {val_false:<12.4f} {val_true:<12.4f} {diff:>+12.4f}")
    
    return results



def main():
    """Main inference function."""
    args = get_inference_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup environment
    setup_environment(args.gpu)
    
    
    # Run inference for each seed
    all_results = []
    total_start_time = time.time()
    
    for seed in args.seeds:
        seed_results = run_inference_single_seed(args.model_path, seed, args)
        all_results.extend(seed_results)  # Each seed returns 2 results now

    
    total_time = time.time() - total_start_time
    
    if not all_results:
        logging.error("No successful inference runs")
        return
    
    # Separate results by permute_ports setting
    results_no_permute = [r for r in all_results if not r['permute_ports']]
    results_permuted = [r for r in all_results if r['permute_ports']]
    
    # Print summary
    print(f"\nInference Summary:")
    print(f"  Seeds processed: {len(args.seeds)}")
    print(f"  Total conditions: {len(all_results)} (2 per seed)")
    print(f"  Total time: {total_time:.2f}s")
    
    # Print aggregated results across seeds for each condition
    if results_no_permute and results_permuted:
        print(f"\n" + "="*70)
        print(f"AGGREGATED RESULTS ACROSS {len(args.seeds)} SEEDS")
        print(f"="*70)
        
        metrics = results_no_permute[0]['metrics'].keys()
        
        print(f"\n{'Metric':<15} {'No Permute':<15} {'Permuted':<15} {'Avg Difference':<15}")
        print("-" * 65)
        
        for metric in metrics:
            # Calculate averages for each condition
            vals_no_permute = [r['metrics'][metric] for r in results_no_permute]
            vals_permuted = [r['metrics'][metric] for r in results_permuted]
            
            avg_no_permute = np.mean(vals_no_permute)
            std_no_permute = np.std(vals_no_permute)
            avg_permuted = np.mean(vals_permuted)
            std_permuted = np.std(vals_permuted)
            avg_diff = avg_permuted - avg_no_permute
            
            print(f"{metric:<15} {avg_no_permute:.4f}±{std_no_permute:.4f}   {avg_permuted:.4f}±{std_permuted:.4f}   {avg_diff:>+.4f}")
        
        print(f"\nDetailed Statistics:")
        print(f"{'Metric':<15} {'Condition':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 75)
        
        for metric in metrics:
            vals_no_permute = [r['metrics'][metric] for r in results_no_permute]
            vals_permuted = [r['metrics'][metric] for r in results_permuted]
            
            print(f"{metric:<15} {'No Permute':<12} {np.mean(vals_no_permute):<10.4f} {np.std(vals_no_permute):<10.4f} {np.min(vals_no_permute):<10.4f} {np.max(vals_no_permute):<10.4f}")
            print(f"{'':<15} {'Permuted':<12} {np.mean(vals_permuted):<10.4f} {np.std(vals_permuted):<10.4f} {np.min(vals_permuted):<10.4f} {np.max(vals_permuted):<10.4f}")
            print()


if __name__ == '__main__':
    main()