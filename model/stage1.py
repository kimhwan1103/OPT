from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2

def dnn(image, frame):
	model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
	image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

	inputs = image_processor(images=image, return_tensors="pt")
	outputs = model(**inputs)

	target_sizes = torch.tensor([image.size[::-1]])
	results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

	confidences = []

	for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
		box = [round(i, 2) for i in box.tolist()]
		confidences = round(score.item(), 3)
		if confidences > 0.5:
			print(f'score : {score} | lable : {label} | box : {box}')
			cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), (0, 255, 0), 2)


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()

	if not ret:
		print("not Frame")
		break

	frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(frame2)
	dnn(image, frame)

	cv2.imshow("TEST", frame)

	key = cv2.waitKey(1)
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
