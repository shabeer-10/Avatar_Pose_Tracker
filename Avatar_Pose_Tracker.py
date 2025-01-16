import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose for motion tracking
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Avatar rendering function
def draw_avatar(frame, pose_landmarks):
    # Create a blank canvas for the avatar
    avatar_frame = np.zeros_like(frame)

    if pose_landmarks:
        for idx, landmark in enumerate(pose_landmarks.landmark):
            # Convert normalized coordinates to pixel values
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Draw key points for arms, legs, and torso
            if idx in [11, 12, 13, 14, 23, 24]:
                cv2.circle(avatar_frame, (x, y), 10, (0, 255, 0), -1)

        # Draw head portion using landmarks for nose and eyes
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

        # Head center and radius
        head_center = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
        head_radius = int(((left_eye.x - right_eye.x) * frame.shape[1]) * 2.5)

        # Draw the head (circle)
        cv2.circle(avatar_frame, head_center, head_radius, (255, 0, 0), -1)

        # Add eyes
        left_eye_center = (int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0]))
        right_eye_center = (int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0]))

        cv2.circle(avatar_frame, left_eye_center, 5, (255, 255, 255), -1)  # Left eye
        cv2.circle(avatar_frame, right_eye_center, 5, (255, 255, 255), -1)  # Right eye

        # Draw connections between joints (basic skeleton)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
        ]
        for connection in connections:
            start_idx, end_idx = connection
            start_landmark = pose_landmarks.landmark[start_idx]
            end_landmark = pose_landmarks.landmark[end_idx]

            start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
            end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))

            cv2.line(avatar_frame, start_point, end_point, (255, 255, 255), 5)

    return avatar_frame


# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose landmarks
        results = pose.process(rgb_frame)

        # Draw the avatar based on pose landmarks
        avatar_image = draw_avatar(frame, results.pose_landmarks)

        # Resize avatar image to match the original frame dimensions
        avatar_image_resized = cv2.resize(avatar_image, (frame.shape[1], frame.shape[0]))

        # Combine original frame and avatar for display
        combined_frame = np.hstack((frame, avatar_image_resized))

        # Display the frames
        cv2.imshow("Avatar Tracking (Left: Original, Right: Avatar)", combined_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()