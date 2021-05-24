import argparse
import importlib
import pytorch_lightning as pl
from text_classifier import lit_models


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_classifer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Add trainer arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument(
        "--wandb", action="store_true", default=False
    )  # when setting the "--wandb" flag the value will be True, if it is omitted the value is False
    parser.add_argument("--data_class", type=str, default="IMDBTransformer")
    parser.add_argument("--model_class", type=str, default="DistilBERTClassifier")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--model_checkpoint", type=str, default=None)

    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_classifier.data.{temp_args.data_class}")
    model_class = _import_class(f"text_classifier.models.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    # model_group = parser.add_argument_group("Model Args")
    # model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    # pl.seed_everything(2401)
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_classifier.data.{args.data_class}")
    model_class = _import_class(f"text_classifier.models.{args.model_class}")
    if args["model_checkpoint"]:
        data = data_class(args, model_checkpoint=args["model_checkpoint"])
        model = model_class(data_config=data.config(), model_checkpoint=args["model_checkpoint"], args=args)
    else:
        data = data_class(args)
        model = model_class(data_config=data.config(), args=args)


    lit_model_class = lit_models.TransformerLitModel

    if args.load_checkpoint is not None:
        # Load model from checkpoint
        pass
    else:
        lit_model = lit_model_class(args=args, model=model)

    # Setup callbacks
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    ]

    args.weights_summary = "full"  # print full summary of the model before training
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=None, default_root_dir="training/logs"
    )

    trainer.tune(
        lit_model, datamodule=data
    )  # if passing --auto_lr_find, this will set learning rate
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()