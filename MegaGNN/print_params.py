import datetime
import logging
import torch
from torch_geometric import seed_everything

import MegaGNN  # noqa, register custom modules
from MegaGNN.graphgym.cmd_args import parse_args
from MegaGNN.graphgym.config import (cfg, dump_cfg, dump_imp_cfg, set_cfg, load_cfg)
from MegaGNN.graphgym.loader import create_loader
from MegaGNN.graphgym.logger import setup_printing
from MegaGNN.graphgym.optimizer import create_optimizer, create_scheduler
from MegaGNN.graphgym.model_builder import create_model
from MegaGNN.graphgym.register import train_dict
from MegaGNN.graphgym.utils.comp_budget import params_count
from MegaGNN.graphgym.utils.device import auto_select_device
from MegaGNN.logger import create_logger
from MegaGNN.utils import (
    new_optimizer_config,
    new_scheduler_config,
    custom_set_out_dir,
    custom_set_run_dir
)

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_config(args):
    """Setup configuration from command line arguments."""
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)


def setup_environment(args):
    """Setup PyTorch environment and device."""
    torch.set_num_threads(cfg.num_threads)
    
    if args.gpu == -1:
        auto_select_device(strategy='greedy')
    else:
        logging.info('Select GPU {}'.format(args.gpu))
        if cfg.device == 'auto':
            cfg.device = f'cuda:{args.gpu}'


def setup_run(run_id=1):
    """Setup specific run configuration."""
    custom_set_run_dir(cfg, run_id=cfg.seed)
    setup_printing()
    seed_everything(cfg.seed)
    
    cfg.cfg_dest = cfg.run_dir + "/config.yaml"
    dump_cfg(cfg)
    dump_imp_cfg(cfg)

    logging.info(f"[*] Run ID {cfg.seed}: seed={cfg.seed}")
    logging.info(f"    Starting now: {datetime.datetime.now()}")


def create_training_components(dataset):
    """Create model, optimizer, and scheduler."""
    model = create_model(dataset=dataset)
    optimizer = create_optimizer(model.named_parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
    
    # Log model information
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    
    return model, optimizer, scheduler


def main():
    args = parse_args()
    setup_config(args)
    setup_environment(args)
    setup_run()

    # Create training components
    loaders, dataset = create_loader(returnDataset=True)
    loggers = create_logger()
    model, optimizer, scheduler = create_training_components(dataset)
    
    print(100*"-")
    print(args.cfg_file)
    print(cfg.gnn.multi_edge_agg)
    print(cfg.gnn.multi_edge_agg_type)
    print(cfg.dataset.reverse_mp)
    print(cfg.train.add_ego_id)
    print(params_count(model))
    print(100*"-")


if __name__ == '__main__':
    main()