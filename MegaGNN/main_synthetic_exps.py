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


def max_port_cols(d):
    """
    Returns (max_in_port_id, max_out_port_id)
    whose edge_attr last two columns correspond to (in_port_id, out_port_id).
    """
    edge_attr = d['node', 'to', 'node'].edge_attr
    in_col, out_col = edge_attr.size(-1) - 2, edge_attr.size(-1) - 1
    return int(edge_attr[:, in_col].max().item()), int(edge_attr[:, out_col].max().item())


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

    if cfg.dataset.add_ports:
        tr_in_max, tr_out_max = max_port_cols(dataset['train'])
        va_in_max, va_out_max = max_port_cols(dataset['val'])
        te_in_max, te_out_max = max_port_cols(dataset['test'])
        in_port_vocab_size  = max(tr_in_max,  va_in_max,  te_in_max)  + 1
        out_port_vocab_size = max(tr_out_max, va_out_max, te_out_max) + 1
        dataset.in_port_vocab_size = in_port_vocab_size
        dataset.out_port_vocab_size = out_port_vocab_size
    else:
        dataset.in_port_vocab_size  = 0
        dataset.out_port_vocab_size = 0

    model, optimizer, scheduler = create_training_components(dataset)

    # Start training
    train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    logging.info(f"[*] All done: {datetime.datetime.now()}")


if __name__ == '__main__':
    main()