# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cv2

import torch

import os,sys,warnings,shutil
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import subprocess, shlex
from tqdm import tqdm
import modules.dataloaders as dataloaders
from modules.GRUModel import GRUModel
from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.network import get_network
from modules.utils import split_video_fixed_time



class estimate_quality():
	def __init__(self,
		segment_length:float=2,
		process_frame_length:int=8,
		adjust:bool=True,
		ffmpeg_path:str="/usr/bin/ffmpeg",
		regressor_model="LIVE_YT_HFR"
	):
		"""
		Estimate Quality of a video
		
		Args:
			segment_length (float): Length of each video segments when the the input video is split. If `segment_length = None`, video is not split.
			process_frame_length (int): No.of frames to processed together while estimating quality of video i.e features extracted from these frames are passed to GRU units. The larger the process_frame_length, the more GPU memory is necessary.
			adjust (bool): If no.of frames in the input video is less than process_frame_length, the model still processes them if `adjust = True`.
			regressor_model (str): Regressor to use. Options: ["LIVE_ETRI", "LIVE_YT_HFR", "YouTube_UGC"]
		"""
		# Weights
		contrique_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/CONTRIQUE_checkpoint25.tar")
		conviqt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/CONVIQT_checkpoint10.tar")
		regressor_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/{}.save".format(regressor_model))
		
		# Gloabl Parameters
		self.segment_length = segment_length
		self.process_frame_length = process_frame_length
		self.adjust = adjust
		self.ffmpeg_path = ffmpeg_path
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


		# Load CONTRIQUE Model
		encoder = get_network('resnet50')
		model = CONTRIQUE_model(None, encoder, 2048)
		model.load_state_dict(torch.load(contrique_path, map_location=self.device.type))
		self.model = model.to(self.device)
		self.model.eval()

		# Load CONVIQT model
		temporal_model = GRUModel(c_in = 2048, hidden_size = 1024, projection_dim = 128, normalize = True, num_layers = 1)
		temporal_model.load_state_dict(torch.load(conviqt_path, map_location=self.device.type))
		self.temporal_model = temporal_model.to(self.device)
		self.temporal_model.eval()

		# Load Regressor Model
		self.regressor = pickle.load(open(regressor_path, 'rb'))
		

	# Extract Frames
	def extract_frames(self, video_path):
		"""
		Args:
			video_path (string): Video path.
		Returns:
			frames (np.array): Numpy array of frames.
		"""
		video = cv2.VideoCapture(video_path)
		success,image = video.read()

		frames = []
		while success:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			frames.append(image)

			success,image = video.read()

		return np.array(frames)
		
	
	def compute_quality(self, input_video_path):
		# load video
		video = self.extract_frames(input_video_path)
		T, height, width, C = video.shape
		
		# Define torch transform for 2 spatial scales
		transform = dataloaders.torch_transform((height, width))
		
		# Define arrays to store frames
		frames = torch.zeros((T,3,height,width), dtype=torch.float16)
		frames_2 = torch.zeros((T,3,height// 2,width// 2), dtype=torch.float16)
		
		# Read every video frame
		for frame_ind in range(T):
			inp_frame = Image.fromarray(video[frame_ind])
			inp_frame, inp_frame_2 = transform(inp_frame)
			frames[frame_ind], frames_2[frame_ind] = inp_frame.type(torch.float16), inp_frame_2.type(torch.float16)
		
		# Convert to torch tensors
		if self.adjust and T < self.process_frame_length:
			batch_size = T
		else:
			batch_size = self.process_frame_length
		loader = dataloaders.create_data_loader(frames, frames_2, batch_size)
		
		# Extract CONTRIQUE features
		video_feat = dataloaders.extract_features(self.model, loader)
		
		# Extract CONVIQT features
		feat_frames = torch.from_numpy(video_feat[:,:2048])
		feat_frames_2 = torch.from_numpy(video_feat[:,2048:])
		loader = dataloaders.create_data_loader(feat_frames, feat_frames_2, batch_size)
		video_feat = dataloaders.extract_features_temporal(self.temporal_model, loader)
		
		# Predicting Score using regressor model
		score = self.regressor.predict(video_feat)[0]
		
		return score
	

	def get_scores(self, input_video_path):
		if self.segment_length is None:
			return self.compute_quality(input_video_path)
		
		# Temp Path
		temp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_video_segments")
		os.makedirs(temp_path, exist_ok=True)

		# Splitting video into segments
		cmd = split_video_fixed_time(
			input_path=input_video_path,
			time_length=self.segment_length,
			output_dir=temp_path,
			ffmpeg_path=self.ffmpeg_path
		)
		subprocess.run(shlex.split(cmd))
		
		# Get and Save Quality Scores
		Scores = []
		for video_file in tqdm(os.listdir(temp_path)):
			Scores.append(self.compute_quality(os.path.join(temp_path, video_file)))
		np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.splitext(os.path.basename(input_video_path))[0] + ".npy"), np.asarray(Scores))

		# Removing temp path
		shutil.rmtree(temp_path)