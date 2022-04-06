""" Restart previous training

Arguments: file_name config.yaml(optional)

file_name: trainer.pth from a previous training
config.yaml: any parameters that needs to be revised
"""
import argparse
import logging

import torch

from estorch.scripts.train import regression_by_species
from nequip.utils import Config, dataset_from_config, load_file


def main(args=None):
    file_name, config = parse_command_line(args)
    restart(file_name, config, mode="update")


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Restart an existing NequIP training session.")
    parser.add_argument("session", help="trainer.pth from a previous training")
    parser.add_argument("--update-config", help="File containing any config paramters to update")
    args = parser.parse_args(args=args)

    if args.update_config:
        config = Config.from_file(args.update_config)
    else:
        config = Config()

    config.append = config.get("append", True)
    if config.append is None:
        config.append = True
    config.wandb_resume = config.get("wandb_resume", True)

    return args.session, config


def restart(file_name, config, mode="update"):

    # load the dictionary
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=file_name,
        enforced_format="torch",
    )

    dictionary.update(config)
    dictionary["run_time"] = 1 + dictionary.get("run_time", 0)

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )

    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB

        # resume wandb run
        if config.wandb_resume:
            from nequip.utils.wandb import resume

            resume(config)
        else:
            from nequip.utils.wandb import init_n_update

            config = init_n_update(config)

        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer.from_dict(dictionary)

    config.update(trainer.output.updated_dict())

    dataset = dataset_from_config(config)
    logging.info(f"Successfully reload the data set of type {dataset}...")

    # subtract energy offset
    if config.get("regression_by_species", False):
        dataset = regression_by_species(dataset, trainer.train_idcs)

    trainer.set_dataset(dataset)
    trainer.train()

    return


if __name__ == "__main__":
    main()
