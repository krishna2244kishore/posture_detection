import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import mediapipe as mp
import base64
from flask_socketio import SocketIO, emit

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_posture_rule_based(landmarks, image_shape, mode='squat'):
    feedback = []
    h, w = image_shape
    # Get required keypoints
    try:
        left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w,
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
        right_hip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
        left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
        right_knee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h]
        left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        right_ankle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
        nose = [landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].x * w,
                landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y * h]
    except Exception as e:
        return ['Keypoints missing']

    if mode == 'squat':
        # Rule 1: Knee over toe
        if left_knee[0] > left_ankle[0] or right_knee[0] > right_ankle[0]:
            feedback.append('Knee over toe detected')
        # Rule 2: Back angle < 150° (shoulder-hip-ankle)
        left_back_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_back_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        if left_back_angle < 150 or right_back_angle < 150:
            feedback.append('Back angle too small (hunched back)')
    elif mode == 'desk':
        # Rule 1: Neck bend > 30° (shoulder-hip-nose)
        left_neck_angle = calculate_angle(left_shoulder, left_hip, nose)
        right_neck_angle = calculate_angle(right_shoulder, right_hip, nose)
        if left_neck_angle > 30 or right_neck_angle > 30:
            feedback.append('Neck bent too much')
        # Rule 2: Back not straight (shoulder-hip-ankle)
        left_back_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_back_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        if left_back_angle < 165 or right_back_angle < 165:
            feedback.append('Back not straight')
    return feedback

def detect_squat_rep(prev_state, left_knee_angle, right_knee_angle, feedback):
    # Define thresholds for squat detection
    down_threshold = 90  # knee angle less than this means 'down'
    up_threshold = 160   # knee angle greater than this means 'up'
    # Only count rep if feedback is empty (perfect form)
    if left_knee_angle < down_threshold and right_knee_angle < down_threshold and prev_state == 'up' and not feedback:
        return 'down', False
    elif left_knee_angle > up_threshold and right_knee_angle > up_threshold and prev_state == 'down' and not feedback:
        return 'up', True
    else:
        return prev_state, False

def detect_pushup_rep(prev_state, left_elbow_angle, right_elbow_angle, feedback):
    # Define thresholds for pushup detection
    down_threshold = 70  # elbow angle less than this means 'down'
    up_threshold = 160   # elbow angle greater than this means 'up'
    if left_elbow_angle < down_threshold and right_elbow_angle < down_threshold and prev_state == 'up' and not feedback:
        return 'down', False
    elif left_elbow_angle > up_threshold and right_elbow_angle > up_threshold and prev_state == 'down' and not feedback:
        return 'up', True
    else:
        return prev_state, False

def calculate_joint_angle(landmarks, a_idx, b_idx, c_idx, w, h):
    a = [landmarks[a_idx].x * w, landmarks[a_idx].y * h]
    b = [landmarks[b_idx].x * w, landmarks[b_idx].y * h]
    c = [landmarks[c_idx].x * w, landmarks[c_idx].y * h]
    return calculate_angle(a, b, c)

def analyze_pose(filepath, mode='squat', sid=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(filepath)
    frame_results = []
    frame_idx = 0
    rep_count = 0
    prev_state = 'up'  # Initial state for rep detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            feedback = analyze_posture_rule_based(landmarks, frame.shape[:2], mode=mode)
            keypoints = [
                {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                } for lm in landmarks
            ]
            h, w = frame.shape[:2]
            if mode == 'squat':
                left_knee_angle = calculate_joint_angle(
                    landmarks,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value,
                    w, h
                )
                right_knee_angle = calculate_joint_angle(
                    landmarks,
                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_KNEE.value,
                    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                    w, h
                )
                prev_state, rep_inc = detect_squat_rep(prev_state, left_knee_angle, right_knee_angle, feedback)
                if rep_inc:
                    rep_count += 1
            elif mode == 'pushup':
                left_elbow_angle = calculate_joint_angle(
                    landmarks,
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    w, h
                )
                right_elbow_angle = calculate_joint_angle(
                    landmarks,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value,
                    w, h
                )
                prev_state, rep_inc = detect_pushup_rep(prev_state, left_elbow_angle, right_elbow_angle, feedback)
                if rep_inc:
                    rep_count += 1
            elif mode == 'desk':
                # For desk posture, count the number of frames with perfect posture
                if not feedback:
                    rep_count += 1
            # Add more exercises here as needed
        else:
            feedback = ['No person detected']
            keypoints = []
        frame_result = {'frame': frame_idx, 'feedback': feedback, 'keypoints': keypoints}
        frame_results.append(frame_result)
        # Emit live feedback if sid is provided
        if sid is not None:
            socketio.emit('video_frame_feedback', {'frame': frame_idx, 'feedback': feedback}, room=sid)
        frame_idx += 1
    cap.release()
    return {'frames': frame_results, 'rep_count': rep_count}

@app.route('/upload', methods=['POST'])
def upload_video():
    print('request.files:', request.files)
    print('request.form:', request.form)
    print('request.content_type:', request.content_type)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    mode = request.form.get('mode', 'squat')  # Default to squat if not provided
    sid = request.form.get('sid')  # SocketIO session id from frontend
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Analyze pose with live feedback
        results = analyze_pose(filepath, mode=mode, sid=sid)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'results': results}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    # Decode base64 image
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        feedback = analyze_posture_rule_based(landmarks, frame.shape[:2])
        keypoints = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
            for lm in landmarks
        ]
    else:
        feedback = ['No person detected']
        keypoints = []
    return jsonify({'feedback': feedback, 'keypoints': keypoints})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port) 