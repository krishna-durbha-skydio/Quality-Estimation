# Importing Libraries
import os

def split_video_fixed_time(
	input_path:str,
	time_length:float,
	output_dir:str,
	ffmpeg_path:str,
):
	"""
	Splitting video into multiple parts with exactly same amount of time.

	Args:
		input_path (str): Path to input video.
		time_length (float): Time length (in seconds) of each part of split.
		output_dir (str): Directory to save parts of videos.
	"""
	# File
	filename, ext = os.path.splitext(os.path.basename(input_path))
	
	# Assertions
	assert ext != ".yuv", "Invalid input video format i.e .yuv file."

	# Command
	cmd = ffmpeg_path
	cmd_split = "-i {} -acodec copy -f segment -segment_time {} -vcodec copy -reset_timestamps 1 -map 0 {}/{}_%d{}".format(input_path, time_length, output_dir, filename, ext)

	cmd = " ".join([cmd, cmd_split])
	
	return cmd