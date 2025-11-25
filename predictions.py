import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


MODEL_PATH = r"gesture_model.h5"
CLASS_NAMES = ['baby', 'deaf', 'dentist', 'die', 'dizzy', 'doctor','ears','hearing','heart','hospital','hurt','medicine','mouth','nose','nurse','sick']  
LANDMARK_SIZE = 225 

model = load_model(MODEL_PATH)

mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)



def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_holistic.process(frame_rgb)

    pose = []
    left_hand = []
    right_hand = []

   
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0] * (33 * 3)  

    
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            left_hand.extend([lm.x, lm.y, lm.z])
    else:
        left_hand = [0] * (21 * 3)  

    
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            right_hand.extend([lm.x, lm.y, lm.z])
    else:
        right_hand = [0] * (21 * 3)  

    
    return np.array(pose + left_hand + right_hand)



def get_sequence(path):
    seq = []
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Invalid path")
        seq.append(extract_landmarks(img))
        return seq

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        seq.append(extract_landmarks(frame))

    cap.release()
    return seq



def prepare_sequence(seq):
    seq = np.array(seq)

    if seq.shape[0] >= MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    else:
        pad = np.zeros((MAX_SEQ_LEN - seq.shape[0], LANDMARK_SIZE))
        seq = np.vstack((seq, pad))

    return np.expand_dims(seq, axis=0)  


def predict_sign(path):
    print("Processing:", path)

    seq = get_sequence(path)
    seq = prepare_sequence(seq)

    pred = model.predict(seq)[0]   

    top3_idx = pred.argsort()[-3:][::-1]  

    print("\n🔮 Top Predictions:")
    for idx in top3_idx:
        print(f"  {CLASS_NAMES[idx]}  —  {pred[idx] * 100:.2f}%")

    
    best_class = CLASS_NAMES[top3_idx[0]]
    best_conf = pred[top3_idx[0]] * 100
    

    return best_class, best_conf



if __name__ == "__main__":


    
    input_path = r"hurt.mp4"
    predict_sign(input_path)
