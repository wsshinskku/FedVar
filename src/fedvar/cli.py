import argparse
import json
from pathlib import Path

from .config import Config
from .training import evaluate_checkpoint, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="FedVar official author implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train one synchronous federated experiment")
    comparison = commands.add_parser(
        "compare", help="Run methods with the same seeded dataset/partition"
    )
    for subparser in (training, comparison):
        subparser.add_argument("--config")
        subparser.add_argument("--output", required=True)
        subparser.add_argument("--seed", type=int)
        subparser.add_argument("--dataset", choices=["synthetic", "mnist", "cifar10", "cifar100"])
        subparser.add_argument("--model")
        subparser.add_argument("--device")
        subparser.add_argument("--clients", type=int)
        subparser.add_argument("--clients-per-round", type=int)
        subparser.add_argument("--rounds", type=int)
        subparser.add_argument("--download", action="store_true", default=None)
    training.add_argument("--algorithm", choices=["fedvar", "fedavg", "fedsgd", "fedprox"])
    comparison.add_argument(
        "--algorithms",
        nargs="+",
        default=["fedavg", "fedsgd", "fedprox", "fedvar"],
        choices=["fedvar", "fedavg", "fedsgd", "fedprox"],
    )
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate the saved server model on the test split"
    )
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_checkpoint(args.checkpoint, args.device)
        text = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    values = vars(args).copy()
    command, path, output = values.pop("command"), values.pop("config"), values.pop("output")
    algorithms = values.pop("algorithms", None)
    config = Config.load(path, **values)
    if command == "train":
        train(config, output)
    else:
        target = Path(output)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError("Comparison output directory must be new or empty")
        if len(set(algorithms)) != len(algorithms):
            raise ValueError("Comparison algorithms must be distinct")
        target.mkdir(parents=True, exist_ok=True)
        results = {}
        for algorithm in algorithms:
            current = Config(**{**config.as_dict(), "algorithm": algorithm})
            result = train(current, target / algorithm)
            results[algorithm] = result["final"]
        (target / "comparison.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
