import os

# stop tensorboard warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# stop W&B logs
os.environ['WANDB_SILENT'] = 'true'

import math
import pprint
import argparse
import warnings

import utils.config_parser

import train
import predict

pp = pprint.PrettyPrinter(indent=1)

def parse_args():
    parser = argparse.ArgumentParser(description='HPA')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    parser.add_argument(
        '--dev', 
        action="store_true", 
        help="Start in dev mode"
    )
    parser.add_argument(
        '-p',
        '--publish', 
        action="store_true",
        help="Publish results on W&B (only in training)"
    )
    parser.add_argument(
        '-v',
        '--vote', 
        action="store_true",
        help="Vote a submission from existing results"
    )
    parser.add_argument(
    	'-mp',
    	'--mixed_precision'
    	action="store_true",
    	help="Train in mixed precision"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file!')

    config = utils.config_parser.load(args.config_file)
    
    config.publish = args.publish
    config.dev = args.dev

    config.vote = args.vote
    config.mp=args.mixed_precision
    # pp.pprint(config)

    if config.mode == "TRA":
        train.run(config)
    elif config.mode == "PRD":
        predict.run(config)

if __name__ == '__main__':
    main()
