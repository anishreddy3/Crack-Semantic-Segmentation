import onnxruntime as rt
import numpy as np
import sklearn
import skl2onnx
from skonnxrt.sklapi import OnnxTransformer
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import imutils
import argparse
from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import time
import onnx
from onnx import optimizer
from threading import Thread


class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()






class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream", resolution=(1080,1080)):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		self.stream.set(3, int(resolution[0]))
		self.stream.set(4, int(resolution[1]))
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")


#LOADING CRACK DETECTING MODEL TRAINED ON PYTORCH CONVERTED TO ONNX FORMAT
model_file = ("/home/aniyo/onnx_model_name.onnx")

print("[INFO] sampling THREADED frames from webcam...")
#vsa = WebcamVideoStream(src=2).start()
vs = cv2.VideoCapture(args["video"])
writer = None
# try to determine the total number of frames in the video file
# try to determine the total number of frames in the video file
try:
	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1
fps = FPS().start()


while True:
	#load the input image and resize it to fit our model requirements
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	#frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_LINEAR)
	frame = imutils.resize(frame, width= 256)

	#constructing a blob from our image
	blob = cv2.dnn.blobFromImage(cv2.resize(
	    frame, (256, 256)), 1/255.0, (256, 256), 0, swapRB=True, crop=False)

	#Open the model and run forward pass with the given blob
	with open(model_file, "rb") as f:
	    model_bytes = f.read()
	ot = OnnxTransformer(model_bytes)
	start = time.time()
	pred = ot.fit_transform(blob)
	end = time.time()

	# Infer the total number of classes, height and width
	(numClasses, height, width) = pred.shape[1:4]

	# Argmax is utilized to find the class label with largest probability for every pixel in the image
	classMap = np.argmax(pred[0], axis=0)

	# classes are mapped to their respective colours
	mask = COLORS[classMap]

	# resizing the mask and class map to match its dimensions with the input image
	mask = cv2.resize(
	    mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
	classMap = cv2.resize(
	    classMap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

	# Construct weighted combination of input image along with mask to form output visualization
	output = ((0.4 * frame) + (0.6 * mask)).astype("uint8")


	# resizing the output display window
	cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Output',800,800)

	# displaying the output
	cv2.imshow("Output",output)
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(output.shape[1], output.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(output)

	# quit the process when q is pressed
	key = cv2.waitKey(1) & 0xff

	if key == ord("q"):
		break

	fps.update()

fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx FPS: {:.2f}".format(fps.fps()))

writer.release()
vs.release()
