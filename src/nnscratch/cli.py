"""nnscratch: train the from-scratch network on digits or moons."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .datasets import digits, moons
from .gradcheck import check_gradients
from .network import mlp
from .optimisers import SGD, Adam

ROOT = Path(__file__).resolve().parents[2]


def _optimiser(name: str, lr: float):
    return Adam(lr) if name == "adam" else SGD(lr, momentum=0.9)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nnscratch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train on the 8x8 digits dataset")
    t.add_argument("--hidden", default="128,64", help="hidden layer sizes, e.g. 128,64")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--optimiser", choices=["sgd", "adam"], default="sgd")
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--charts", type=Path, default=None)
    t.add_argument("--save", type=Path, default=None)

    m = sub.add_parser("moons", help="train on two half-moons (2 inputs, 2 classes)")
    m.add_argument("--epochs", type=int, default=60)
    m.add_argument("--lr", type=float, default=0.1)

    c = sub.add_parser("compare", help="SGD against Adam on the same data and seed")
    c.add_argument("--epochs", type=int, default=25)

    sub.add_parser("gradcheck", help="verify backprop against numerical gradients")

    args = ap.parse_args(argv)

    if args.cmd == "gradcheck":
        rng = np.random.default_rng(0)
        worst = 0.0
        for sizes in ([4, 6, 3], [3, 8, 8, 2], [5, 4, 4]):
            net = mlp(sizes, dropout=0.0, seed=1)
            x = rng.normal(size=(6, sizes[0]))
            y = np.eye(sizes[-1])[rng.integers(0, sizes[-1], 6)]
            err = check_gradients(net, x, y)
            print(f"{'-'.join(map(str, sizes)):<12} worst relative error {err:.2e}")
            worst = max(worst, err)
        print("backprop matches numerical gradients" if worst < 1e-5 else "MISMATCH: check backward()")
        return 0 if worst < 1e-5 else 1

    if args.cmd == "moons":
        x_train, x_test, y_train, y_test = moons()
        net = mlp([2, 16, 16, 2], seed=0, optimiser=SGD(args.lr, momentum=0.9))
        net.fit(x_train, y_train, epochs=args.epochs, batch_size=16, validation=(x_test, y_test))
        print(f"train accuracy {net.score(x_train, y_train):.3f}   test accuracy {net.score(x_test, y_test):.3f}")
        print("A network with no hidden layer cannot solve this; ours reaches the high 90s.")
        return 0

    if args.cmd == "compare":
        x_train, x_test, y_train, y_test = digits()[:4]
        print(f"{'optimiser':<10} {'epochs':>7} {'train acc':>10} {'test acc':>9} {'seconds':>8}")
        for name, lr in (("sgd", 0.05), ("adam", 0.005)):
            net = mlp([64, 128, 64, 10], seed=0, optimiser=_optimiser(name, lr))
            start = time.perf_counter()
            net.fit(x_train, y_train, epochs=args.epochs, batch_size=32)
            print(f"{name:<10} {args.epochs:>7} {net.score(x_train, y_train):>10.3f} "
                  f"{net.score(x_test, y_test):>9.3f} {time.perf_counter() - start:>8.1f}")
        return 0

    x_train, x_test, y_train, y_test, test_images = digits(seed=args.seed)
    sizes = [x_train.shape[1], *[int(h) for h in args.hidden.split(",") if h], y_train.shape[1]]
    net = mlp(sizes, dropout=args.dropout, seed=args.seed, optimiser=_optimiser(args.optimiser, args.lr))
    print(f"architecture {'-'.join(map(str, sizes))}, {args.optimiser}, lr {args.lr}, "
          f"dropout {args.dropout}, {len(x_train)} training samples")
    start = time.perf_counter()
    net.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
            validation=(x_test, y_test), seed=args.seed, verbose=True, early_stopping=8)
    seconds = time.perf_counter() - start
    train_acc, test_acc = net.score(x_train, y_train), net.score(x_test, y_test)
    print(f"\ntrained in {seconds:.1f}s   train accuracy {train_acc:.4f}   test accuracy {test_acc:.4f}")

    y_true = y_test.argmax(axis=1)
    y_pred = net.predict(x_test)
    wrong = int((y_true != y_pred).sum())
    print(f"{wrong} wrong out of {len(y_true)} test digits")

    if args.save:
        net.save(args.save)
        print(f"weights saved to {args.save}")
    if args.charts:
        from .plots import confusion, learning_curves, mistakes
        args.charts.mkdir(parents=True, exist_ok=True)
        print("chart:", learning_curves(net.history, args.charts / "learning-curves.png"))
        print("chart:", confusion(y_true, y_pred, args.charts / "confusion-matrix.png"))
        p = mistakes(test_images, y_true, y_pred, args.charts / "mistakes.png")
        if p:
            print("chart:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
