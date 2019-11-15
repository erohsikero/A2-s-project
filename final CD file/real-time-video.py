import numpy as np
import time
import cv2 as cv2
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment
AudioSegment.converter = "C:/Users/annu1/Desktop/kklab/A2's_PROJECT/yolo-master/ffmpeg-20190927-0485865-win64-static/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "C:/Users/annu1/Desktop/kklab/A2's_PROJECT/yolo-master/ffmpeg-20190927-0485865-win64-static/bin/ffmpeg.exe"
AudioSegment.ffprobe ="C:/Users/annu1/Desktop/kklab/A2's_PROJECT/yolo-master/ffmpeg-20190927-0485865-win64-static/bin/ffprobe.exe"


# load the COCO class labels our YOLO model was trained on
LABELS = open("./yolo-coco/coco.names").read().strip().split("\n")


# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("./yolo-coco/yolov3.cfg", "./yolo-coco/yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(type(cap))
frame_count = 0
start = time.time()
first = True
frames = []
colours = np.random.randint(0,255, size = (len(LABELS),3), dtype = "uint8")


while True:
	frame_count += 1
    # Capture frame-by-frameq
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	frames.append(frame)

	if ret:
		key = cv2.waitKey(1)
		if frame_count % 60 == 0:
			end = time.time()
			# grab the frame dimensions and convert it to a blob
			(H, W) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []
			centers = []

			for output in layerOutputs:
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions 
					if confidence > 0.5:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))

			# apply non-maxima suppression to suppress weak, overlapping bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

			texts = []

			# ensure at least one detection exists
			if len(idxs) > 0:
			
				for i in idxs.flatten():
					centerX, centerY = centers[i][0], centers[i][1]
					(x,y) = (boxes[i][0], boxes[i][1])
					(w,h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in colours[classIDs[i]]]
					
					cv2.rectangle(frame, (x,y), (x + w, y + h), color , 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
					cv2.putText(frame , text, (x, y-5) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)
					
					if centerX <= W/3:
						W_pos = "left "
					elif centerX <= (W/3 * 2):
						W_pos = "center "
					else:
						W_pos = "right "
					
					if centerY <= H/3:
						H_pos = "top "
					elif centerY <= (H/3 * 2):
						H_pos = "mid "
					else:
						H_pos = "bottom "

					texts.append(H_pos + W_pos + LABELS[classIDs[i]])

			print(texts)
			imS = cv2.resize(frame, (960, 540))
			cv2.imshow('frame',imS)

			k=cv2.waitKey(1)
			if k ==ord('q'):
				break

cap.release()
cv2.destroyAllWindows()
