import argparse
import os

import sys
from pathlib import Path

root = Path(__file__).parent / "dex_ycb_toolkit"
sys.path.insert(0, str(root))

data_root = Path(__file__).resolve().parent.parent.parent.parent / "data" / "dex_ycb"
os.environ["DEX_YCB_DIR"] = str(data_root)

print(os.environ["DEX_YCB_DIR"])

from dex_ycb_toolkit.hpe_eval import HPEEvaluator
from hmm_utils import hmm_eval, hmm_train, hmm_val_temp

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run HPE evaluation.")
    parser.add_argument("--name", help="Dataset name", default=None, type=str)
    parser.add_argument(
        "--res_file", help="Path to result file", default=None, type=str
    )
    parser.add_argument(
        "--out_dir", help="Directory to save eval output", default=None, type=str
    )
    args = parser.parse_args()
    return args

def main():
    # args = parse_args()

    # args.name = "s0_train"
    # args.res_file = os.path.join(
    #     os.path.dirname(__file__),
    #     "results",
    #     "hpe_spurr_resnet50_{}.txt".format(args.name),
    #     #"example_results_hpe_{}.txt".format(args.name),
    # )

    # if args.name is None and args.res_file is None:
    #     args.name = "s0_train"
    #     args.res_file = os.path.join(
    #         os.path.dirname(__file__),
    #         "results",
    #         "example_results_hpe_{}.txt".format(args.name),
    #    )

    print("hmm_train begin")
    hmm_train("s0_train", "hmm_model_s0_train", override=True)
    print("hmm_train done")
    # val_file_path = os.path.join(
    #         os.path.dirname(__file__),
    #         "results",
    #         "example_results_hpe_{}.txt".format(args.name),
    # )

    print("hmm_val begin")
    hmm_val_temp("s0_val",              
                 model_file_name = "hmm_model_s0_train",
                 out_file = os.path.join(
                    os.path.dirname(__file__),
                    "results",
                    "hmm_val_results_hpe_s0_val.txt"
                ))
    print("hmm_val done")
    # hmm_eval(args.name, 
    #          model_file_name=f"hmm_model_{args.name}", 
    #          res_file = args.res_file, 
    #          out_file = os.path.join(
    #             os.path.dirname(__file__),
    #             "results",
    #             "hmm_results_hpe_{}.txt".format(args.name)
    #         ),
    # )
    
    # hpe_eval = HPEEvaluator(args.name)
    # hpe_eval.evaluate(args.res_file, out_dir=args.out_dir)


if __name__ == "__main__":
    main()