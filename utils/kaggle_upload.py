import subprocess
from os import (walk, listdir, path)
from pathlib import Path
from easydict import EasyDict as edict
import inquirer
import json

from utils.paths import RESULTS_ROOT_DIR


def upload():
	parent_result_dir = Path(RESULTS_ROOT_DIR)

	upload_dir_list = []

	# Check if meta-data already exists (hence already uploaded)
	for dir_name in [dir for dir in listdir(parent_result_dir)]:
		sub_dir = parent_result_dir / dir_name / "dataset-metadata.json"
		if not path.exists(sub_dir):
			upload_dir_list.append(parent_result_dir / dir_name)

	# Ask user which dir(s) to upload
	questions = [
		inquirer.Checkbox('dir_selection',
						  message="Please, select dir(s) to upload?",
						  choices=upload_dir_list,
						  ),
	]
	answers = inquirer.prompt(questions)

	# populate meta-data
	for dir in answers["dir_selection"]:
		session_name = "cornell-birdcall-" + str(dir).split("/")[1].replace("_", "-")

		meta = {
			"title": session_name,
			"id": "humblediscipulus/{}".format(session_name),
			"licenses": [
				{
					"name": "CC0-1.0"
				}
			]
		}

		with open(dir / "dataset-metadata.json", "w") as json_file:
			json_file.write(json.dumps(meta))

	# start uploading
	for dir in answers["dir_selection"]:
		print("[ Uploading dir : {} ]".format(str(dir)))

		p = subprocess.run(["kaggle", "datasets", "create", "-p", str(dir)])
