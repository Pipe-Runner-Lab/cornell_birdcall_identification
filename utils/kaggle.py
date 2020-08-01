import subprocess
from utils.paths import RESULTS_ROOT_DIR
from os import (makedirs, path, environ)
import yaml
import json
from easydict import EasyDict as edict
import os

def kagupload(expname):
	weightspath=path.join(RESULTS_ROOT_DIR,expname)
	metadatapath=path.join(RESULTS_ROOT_DIR,expname,"dataset-metadata.json")
	if not path.exists(weightspath):
		raise Exception('Model weights not unavailable.')
	if not path.exists(metadatapath):
		print("creating a meta data file for uploading")
		subprocess.run(["kaggle", "datasets", "init"],cwd=weightspath)
		with open(metadatapath, 'r') as fid:
			yml_metadata = edict(yaml.safe_load(fid))
		print("writing in the metadata file...")
		yml_metadata.title=expname+"weights"
		url=yml_metadata.id.split('/')[0]
		yml_metadata["id"]=url+'/'+expname+"weights"
		yml_metadata["id"]=yml_metadata["id"].replace("_","-")
		with open(metadatapath, 'w') as json_file:
			json.dump(yml_metadata, json_file)
		num = input ("Uploading data to kaggle, press [Y/N]:")
		if num=="Y" or "y":
			print("starting to upload...")
			pth = os.getcwd()
			dr=path.join(pth,"__results__",expname)
			p=subprocess.run(["kaggle", "datasets", "create","-p",dr],cwd=weightspath, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			print("success")
			return p
		else:
			print("Not uploading.")
	else:
		print("meta-data file exists.Weights already uploaded")

