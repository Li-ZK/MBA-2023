import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch

from config import parse_args
from dataloader import get_dataloaders
from models import model_loader
from utils import get_logger, get_callbacks
from yaml_utils import save_dict_to_yaml


def main(args):
    """
    dls = {
        'train': train_loader,
        'test': test_loader
    }
    :param args:
    :return:
    """
    dls, data_info = get_dataloaders(args)
    # Model
    model = model_loader.get_model(args, data_info)

    # Callbacks and logger
    callbacks = get_callbacks(args)
    tb_logger = get_logger(args)

    # Trainer
    if args.mode in ['train', 'training']:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            # accelerator="cpu",
            callbacks=callbacks,
            num_sanity_val_steps=0,
            logger=tb_logger,
            check_val_every_n_epoch=1,  # verify every n epochs
            enable_checkpointing=False,  # enable checkpoint
            gpus=[0, ],
            # devices='auto',
            # log_every_n_steps=10,
            # auto_select_gpus=True,
        )

        # Fit
        trainer.fit(
            model=model,
            train_dataloaders=dls['train'],
            val_dataloaders=dls['test'],
        )

        # Show training result
        record = model.record
        model_result, model_best_result, res_info = record.print_train_result()

        tb_logger.log_hyperparams({'res_info': str(model_result)})
        tb_logger.log_hyperparams({'res_info': str(model_best_result)})

        print(model_result)
        return model_result, model_best_result, res_info

    elif args.mode in ['test', 'testing']:
        trainer = pl.Trainer(
            accelerator="auto",
            callbacks=callbacks,
            logger=tb_logger,
        )

        # Test
        trainer.test(model=model, dataloaders=dls['test'])
    else:
        raise Exception(f'Error. Mode "{args.mode}" is not supported.')


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Parse args
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    # Path to save experimental results
    yaml_path = 'logger/result_MBA/test.yaml'
    txt_path = 'logger/result_MBA/test.txt'

    is_select_seed = True
    if not is_select_seed:
        # Main
        # setup_seed(1300)
        model_result, model_best_result, res_info = main(args)
    else:
        # save args to yaml file
        save_dict_to_yaml(args, yaml_path)

        # seeds = [1300, 1301, 1303, 1304, 1305, 1306, 1308, 1309, 1313, 1317]
        seeds = [1300]

        for seed in seeds:
            setup_seed(seed)
            args.seed = seed
            model_result, model_best_result, res_info = main(args)
            result = {
                'seed_' + str(seed): {
                    'seed': seed,
                    'model_best_result': model_best_result,
                    'model_result': model_result,
                }
            }

            save_dict_to_yaml(result, yaml_path)

            with open(txt_path, 'a+') as f:
                f.write(args.model + "\n")
                f.write("seed :" + str(seed) + "\n")
                f.write("epochs :" + str(args.epochs) + "\n")
                f.write(res_info)
                f.write("\n\n")
