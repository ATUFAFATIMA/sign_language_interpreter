
import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse


MAX_SEQ_LEN = 30
EPOCHS = 90
BATCH_SIZE = 32


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def extract_keypoints(frame):
    """Extracts holistic keypoints from a frame."""
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        results = holistic.process(frame)
        pose = []
        left_hand = []
        right_hand = []

        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 3)

        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand = np.zeros(21 * 3)

        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            right_hand = np.zeros(21 * 3)

        keypoints = np.concatenate([pose, left_hand, right_hand])
        return keypoints




def preprocess_dataset(dataset_dir, out_dir):
    dataset_dir = r"trimmed_videos"
    out_dir = r"preprocess"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    classes = os.listdir(dataset_dir)

    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        save_cls = os.path.join(out_dir, cls)
        os.makedirs(save_cls, exist_ok=True)

        for vid in os.listdir(cls_path):
            vid_path = os.path.join(cls_path, vid)
            cap = cv2.VideoCapture(vid_path)
            seq = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                kp = extract_keypoints(frame)
                seq.append(kp)

                if len(seq) >= MAX_SEQ_LEN:
                    break

            seq = np.array(seq)
            if seq.shape[0] < MAX_SEQ_LEN:
                padding = np.zeros((MAX_SEQ_LEN - seq.shape[0], seq.shape[1]))
                seq = np.vstack([seq, padding])

            np.save(os.path.join(save_cls, vid.replace('.mp4', '.npy')), seq)





def build_lstm_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Masking(mask_value=0., input_shape=(MAX_SEQ_LEN, input_dim)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model



def load_preprocessed_data(preprocess_dir):
    X = []
    y = []
    class_map = {}
    classes = sorted(os.listdir(preprocess_dir))

    for idx, cls in enumerate(classes):
        class_map[idx] = cls
        cls_path = os.path.join(preprocess_dir, cls)
        for f in os.listdir(cls_path):
            X.append(np.load(os.path.join(cls_path, f)))
            y.append(idx)

    return np.array(X), np.array(y), len(classes), class_map


def train_model(preprocess_dir, model_out, epochs, batch_size, model_type):
    
    preprocess_dir = r"preprocess"
    model_out = r"gesture_model.h5"

    X, y, num_classes, class_map = load_preprocessed_data(preprocess_dir)

    input_dim = X.shape[2]

    if model_type == "lstm":
        model = build_lstm_model(input_dim, num_classes)
    else:
        model = models.Sequential([
            layers.Flatten(input_shape=(MAX_SEQ_LEN, input_dim)),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.4)
    model.save(model_out)
    print("Model saved at:", model_out)





parser = argparse.ArgumentParser(description="Sign Language Pipeline")
sub = parser.add_subparsers(dest='command')


p_pre = sub.add_parser('preprocess')
p_pre.add_argument('--dataset_dir', required=True)   
p_pre.add_argument('--out_dir', required=True)       
p_pre.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN)


p_train = sub.add_parser('train')
p_train.add_argument('--preprocess_dir', required=True)   
p_train.add_argument('--model_out', required=True)        
p_train.add_argument('--epochs', type=int, default=EPOCHS)
p_train.add_argument('--batch_size', type=int, default=BATCH_SIZE)
p_train.add_argument('--model_type', choices=['lstm', 'dense'], default='lstm')

args = parser.parse_args()



if args.command == 'preprocess':
    MAX_SEQ_LEN = args.max_seq_len
    preprocess_dataset(args.dataset_dir, args.out_dir)

elif args.command == 'train':
    train_model(args.preprocess_dir, args.model_out, args.epochs, args.batch_size, args.model_type)
