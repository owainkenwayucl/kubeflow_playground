# Naïve attempt convert the training loop from https://github.com/owainkenwayucl/ML_Playground/tree/main/MedMNIST_PL/General
# This is a terrible way to do it and it needs to be re-arched.

# Needs two PVCs - one to cache dataset ("pathmnist") and one to hold our output ("medmnistcheckpoints")
from kfp.client import Client
import argparse

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('--epochs', metavar='epochs', type=int, help="Set the number of epochs.", default=10)
parser.add_argument('--repeats', metavar='repeats', type=int, help="Set the number of repeats.", default=1)
parser.add_argument('--batch-size', metavar='batchsize', type=int, help="Set the batch size.", default=1024)
parser.add_argument('--base-model', metavar='base_model', type=str, help="Model to use (default resnet18).", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wideresnet50", "wideresnet101", "vgg11"])
args = parser.parse_args()
client = Client()
run = client.create_run_from_pipeline_package(
    'pathmnist_pipeline.yaml',
    arguments={
        "num_epochs": args.epochs,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "base": args.base_model
    },
)