import cv2
import numpy as np
import tensorflow as tf 
from keras.models import load_model

#가중치 파일 
model = load_model('가중치 파일을 넣어주세요')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
	ret, frame = cap.read()

	input_image = cv2.resize(frame, (140, 140))
	input_image = np.expand_dims(input_image, axis=0)
	input_image = input_image / 255.0

	pred = model.predict(input_image)

	label = np.argmax(pred)
	if label == 0:
		cv2.putText(frame, 'person', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	else:
		cv2.putText(frame, 'Not-person', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow("test", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 

cap.release()
cv2.destroyAllWindows()