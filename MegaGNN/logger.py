import logging
import time
import os
import subprocess

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix, precision_recall_curve, auc, classification_report
from sklearn.metrics import r2_score
from MegaGNN.graphgym.config import cfg
from MegaGNN.graphgym.logger import infer_task, Logger
from MegaGNN.graphgym.utils.io import dict_to_json, dict_to_tb
from torchmetrics.functional import auroc

def get_current_gpu_usage():
    '''
    Get the current GPU memory usage.
    '''
    if cfg.gpu_mem and cfg.device != 'cpu' and torch.cuda.is_available():
        result = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        current_pid = os.getpid()
        used_memory = 0
        for line in result.strip().split('\n'):
            line = line.split(', ')
            if current_pid == int(line[0]):
                used_memory += int(line[1])
        return used_memory
    else:
        return -1


def compute_minority_f1_score_per_task(probs, labels, threshold=0.5):

    preds = (probs > threshold)
    y = labels.bool()

    N, C = y.shape
    f1_scores = torch.zeros(C, dtype=torch.float32, device=probs.device)
    epsilon = 1e-12
    
    for c in range(C):
        y_c = y[:, c]

        # Find the minority class (either 0 or 1)
        pos = y_c.sum()
        neg = y_c.numel() - pos
        minority_is_one = (pos <= neg) 

        if minority_is_one:
            y_pos    = y_c
            pred_pos = preds[:, c]
        else:
            y_pos    = ~y_c
            pred_pos = ~preds[:, c]

        true_pos = (y_pos & pred_pos).sum().float()
        false_pos = ((~y_pos) & pred_pos).sum().float()
        false_neg = (y_pos & (~pred_pos)).sum().float()
        
        precision = true_pos / (true_pos + false_pos + epsilon)
        recall = true_pos / (true_pos + false_neg + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores[c] = f1

    return f1_scores.detach().cpu().tolist()

def create_logger():
    """
    Create logger for the experiment

    Returns: List of logger objects

    """
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(CustomLogger(name=names[i], task_type=infer_task()))
    
    return loggers

    

class CustomLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False

    # basic properties
    def basic(self):
        stats = {
            'loss': round(self._loss / self._size_current, max(8, cfg.round)),
            'lr': round(self._lr, max(8, cfg.round)),
            'params': self._params,
            'time_iter': round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats
    
    # task properties
    def classification_binary(self):
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        if true.shape[0] < 1e7:  # AUROC computation for very large datasets is too slow.
            # TorchMetrics AUROC on GPU if available.
            auroc_score = auroc(pred_score.to(torch.device(cfg.device)),
                                true.to(torch.device(cfg.device)),
                                task='binary')
            if self.test_scores:
                # SK-learn version.
                try:
                    r_a_score = roc_auc_score(true.cpu().numpy(),
                                              pred_score.cpu().numpy())
                except ValueError:
                    r_a_score = 0.0
                assert np.isclose(float(auroc_score), r_a_score)
        else:
            auroc_score = 0.
        
        precisions, recalls, _ = precision_recall_curve(true.cpu().numpy(), pred_score.cpu().numpy()) # probs: probabilities for the positive class
        pr_auc = auc(recalls, precisions)
        
        reformat = lambda x: round(float(x), cfg.round)
        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'precision': reformat(precision_score(true, pred_int)),
            'recall': reformat(recall_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int)),
            'macro-f1': reformat(f1_score(true, pred_int, average='macro')),
            'micro-f1': reformat(f1_score(true, pred_int, average='micro')),
            'auc': reformat(auroc_score),
            'pr-auc': reformat(pr_auc)
        }

        return res

    def classification_multi(self):
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        
        # For multiclass, compute metrics with different averaging strategies
        reformat = lambda x: round(float(x), cfg.round)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true.cpu().numpy(), pred_int.cpu().numpy())
        
        # Get per-class metrics using classification_report
        true_np = true.cpu().numpy()
        pred_int_np = pred_int.cpu().numpy()
        report = classification_report(true_np, pred_int_np, output_dict=True)
        
        # Get the list of unique classes
        unique_classes = sorted(list(set(true_np) | set(pred_int_np)))
        
        # Add per-class metrics to results
        per_class_metrics = {}
        for class_idx in unique_classes:
            class_name = f"class_{class_idx}"
            if str(class_idx) in report:  # Report uses string keys
                metrics = report[str(class_idx)]
                per_class_metrics[f'{class_name}_precision'] = reformat(metrics['precision'])
                per_class_metrics[f'{class_name}_recall'] = reformat(metrics['recall'])
                per_class_metrics[f'{class_name}_f1'] = reformat(metrics['f1-score'])
                per_class_metrics[f'{class_name}_support'] = int(metrics['support'])
        
        # Compute multiclass ROC AUC if the dataset is not too large
        if true.shape[0] < 1e7:
            # TorchMetrics AUROC on GPU if available for multiclass
            auroc_score = auroc(pred_score.to(torch.device(cfg.device)),
                               true.to(torch.device(cfg.device)),
                               task='multiclass',
                               num_classes=pred_score.shape[1])
        else:
            auroc_score = 0.
        
        # Create the metrics dictionary with overall metrics
        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'precision_macro': reformat(precision_score(true, pred_int, average='macro')),
            'precision_micro': reformat(precision_score(true, pred_int, average='micro')),
            'precision_weighted': reformat(precision_score(true, pred_int, average='weighted')),
            'recall_macro': reformat(recall_score(true, pred_int, average='macro')),
            'recall_micro': reformat(recall_score(true, pred_int, average='micro')),
            'recall_weighted': reformat(recall_score(true, pred_int, average='weighted')),
            'f1_macro': reformat(f1_score(true, pred_int, average='macro')),
            'f1_micro': reformat(f1_score(true, pred_int, average='micro')),
            'f1_weighted': reformat(f1_score(true, pred_int, average='weighted')),
            'auc': reformat(auroc_score),
        }
        
        # Add per-class metrics if the number of classes is not too large (to avoid excessive logging)
        if len(unique_classes) <= 20:  # Limit to 20 classes for readability
            res.update(per_class_metrics)
        else:
            logging.info(f"Too many classes ({len(unique_classes)}) to log individual class metrics")
            
        # Print the full classification report for more detailed inspection
        logging.info(f"\nClassification Report:\n{classification_report(true_np, pred_int_np)}")
            
        return res
    


    def classification_binary_multiclass(self):
        
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        f1_scores = compute_minority_f1_score_per_task(pred_score, true)
        f1_arr = np.array(f1_scores) * 100

        return {
            'f1': np.round(f1_arr, 4).tolist(),
            'macro_f1': float(np.round(np.mean(f1_scores) * 100, 4)),
        }

    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)
        return {
            'mae': reformat(mean_absolute_error(true, pred)),
            'r2': reformat(r2_score(true, pred, multioutput='uniform_average')),
            'mse': reformat(mean_squared_error(true, pred)),
            'rmse': reformat(mean_squared_error(true, pred, squared=False)),
        }
    
    def update_stats(self, true, pred, loss, lr, time_used, params, index, **kwargs):
        
        assert true.shape[0] == pred.shape[0]
        batch_size = true.shape[0]

        self._iter += 1
        self._index.append(index)
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size
    
    def write_epoch(self, cur_epoch):
        start_time = time.perf_counter()
        basic_stats = self.basic()

        if self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        elif self.task_type == 'classification_binary_multiclass':
            task_stats = self.classification_binary_multiclass()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch,
                       'time_epoch': round(self._time_used, cfg.round)}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round),
                     'eta_hours': round(self.eta(cur_epoch) / 3600, cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()
        if cur_epoch < 3:
            logging.info(f"...computing epoch stats took: "
                         f"{time.perf_counter() - start_time:.2f}s")
        return stats

