from eval import _run

import argparse

if __name__ == "__main__":

    def parse_args():

        parser = argparse.ArgumentParser(description="TODO")

        parser.add_argument(
            "--experiment",
            type=str,
            required=True,
            help="Path to the experiment",
        )
        parser.add_argument(
            "--test_oracle",
            type=str,
            default=None,
            required=True,
            help="Path to test oracle"
        )
        parser.add_argument(
            "--savedir", type=str, default="/mnt/home/viz/design_bench_internal/tmp"
        )
        parser.add_argument("--datadir", type=str, default="/mnt/public/datasets")
        parser.add_argument("--outfile", type=str, default="stats.pkl")
        parser.add_argument("--model", type=str, default="model.pth")
        parser.add_argument("--n_iters", type=int, default=100000)
        parser.add_argument("--gamma", type=float, default=0.0)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--prior_std", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()

    _run(args.experiment, args.savedir, "bprop-latents", args.seed, args)
