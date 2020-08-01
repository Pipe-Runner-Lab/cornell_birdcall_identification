import argparse
from utils.kaggle import kagupload


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_file',
                        help='configuration to upload',
                        default=None, type=str)
	return parser.parse_args()


def main():
	args = parse_args()
	if args.config_file is None:
		raise Exception('no configuration file!')
	p=kagupload(args.config_file.split(".")[0])
	print(p)




if __name__ == '__main__':
    main()