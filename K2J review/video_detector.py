import cv2 as cv
import numpy as np
import time


CONFIDENCE = 0.5
THRESHOLD = 0.3


weights_path = "/root/Desktop/v/yolo-cust/yolov3.weights"
config_path = "/root/Desktop/v/yolo-cust/yolov3.cfg"
label_path = "/root/Desktop/v/yolo-cust/cust.names"
Labels = open(label_path).read().strip().split("\n")
motor_count = 0


#colors
np.random.seed(42)
colours = np.random.randint(0,255, size = (len(Labels),3), dtype = "uint8")

print("[Message]Loading the yolo detector")
net = cv.dnn.readNetFromDarknet(config_path,weights_path)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]


cap = cv.VideoCapture("/root/Desktop/v/saraele.mp4")
#cap = cv.VideoCapture("http://192.168.1.7:8080/video")
writer = None
(W,H) = (None,None)
count =0

try:
	length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	print("[Message] {} total number of frames in the video".format(length)) 
	
except:
	print("[Message] Could'nt determine the number of frames in the video")
	total = -1

while True:
	ret, frame = cap.read()
	count += 10
	#cap.set(1,count)
	
	if not ret:
		break
	
	if W is None or H is None:
		(H,W) = frame.shape[:2]
		print(frame.shape)
	
	blob = cv.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
	# forward the blob to the network
	net.setInput(blob)
	
	
	layerOutputs = net.forward(ln)
	boxes = []
	confidences =[]
	classIDs = []
	
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			
			if confidence > CONFIDENCE:
				box = detection[0:4] * np.array([W,H,W,H])
				(centerX, centerY, width, height) = box.astype("int")
				
				x = int(centerX - (width/2))
				y = int(centerY - (height/2))
				
				boxes.append( [x,y,int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
	idxs = cv.dnn.NMSBoxes(boxes, confidences , CONFIDENCE , THRESHOLD)
			
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x,y) = (boxes[i][0], boxes[i][1])
			(w,h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in colours[classIDs[i]]]
			
			
			if Labels[classIDs[i]]== 'elephant' or Labels[classIDs[i]]== 'tiger':
				cv.rectangle(frame, (x,y), (x + w, y + h), color , 2)
				text = "{}: {:.4f}".format(Labels[classIDs[i]],confidences[i])
				cv.putText(frame , text, (x, y-5) ,  cv.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)
				cv.circle(frame ,(1200,50),20,(0,0,255),-1)		
											
	cv.imshow('frame' , frame)
	k=cv.waitKey(1)
	if k ==ord('q'):
		break
	
	if writer is None:
		# initialize our video writer
		fourcc = cv.VideoWriter_fourcc(*"MJPG")
		writer = cv.VideoWriter('output/op.mp4', fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
	writer.write(frame)
	
	if k==ord('q'):
		break
print("[Message] {} Animals detected".format(motor_count))
cv.destroyAllWindows()
