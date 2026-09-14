import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math


DATA_FILE = "data.csv"

COLUMNS = [
    "neck_angle",
    "neck_mode",
    "knee_angle",
    "knee_mode",
    "timestamp",
    "shldr_diff",
    "hip_diff",
    "angle_l_leg",
    "angle_r_leg",
    "angle_l_arm",
    "angle_r_arm"
]


# ============================================================
# 2. Data Storage Pipeline
# ============================================================

class DataStorage:

    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()

    def load_data(self):
        """
        Pipeline：讀取既有 CSV。

        如果 CSV 不存在：
            建立空 DataFrame。
        """

        try:
            return pd.read_csv(
                self.filename,
                encoding="utf-8"
            )

        except FileNotFoundError:
            return pd.DataFrame(
                columns=COLUMNS
            )

    def create_record(self):
        """
        建立一筆新的分析資料。

        這裡先建立空資料，
        後面的 Analysis Pipeline 再填入結果。
        """

        return {
            column: None
            for column in COLUMNS
        }

    def save(self, result):
        """
        將分析結果存進 DataFrame。
        """

        self.data.loc[len(self.data)] = result

    def write(self):
        """
        最後才真正寫入 CSV。
        """

        self.data.to_csv(
            self.filename,
            index=False,
            encoding="utf-8"
        )


# ============================================================
# 3. Camera Pipeline
# ============================================================

class Camera:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

    def read(self):
        """
        從 Camera 取得一張 Frame。
        """

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):
        """
        釋放 Camera。
        """

        self.cap.release()


# ============================================================
# 4. Pose Detection Pipeline
# ============================================================

class PoseDetector:

    def __init__(self):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def detect(self, frame):
        """
        Pipeline：

        BGR Frame
            ↓
        RGB Frame
            ↓
        MediaPipe
            ↓
        Pose Landmarks
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(
            rgb_frame
        )

        if not results.pose_landmarks:
            return None

        return results.pose_landmarks.landmark


# ============================================================
# 5. Landmark Extraction Pipeline
# ============================================================

class LandmarkExtractor:

    def extract(self, landmarks, frame):

        height, width, _ = frame.shape

        """
        MediaPipe 的座標是 normalized coordinate：

            x = 0 ~ 1
            y = 0 ~ 1

        轉換成 OpenCV pixel coordinate。
        """

        points = {}

        for index, landmark in enumerate(landmarks):

            points[index] = [
                landmark.x * width,
                landmark.y * height
            ]

        return points


# ============================================================
# 6. Posture Analysis
# ============================================================

class PostureAnalyzer:

    # --------------------------------------------------------
    # 顏色
    # --------------------------------------------------------

    COLOR_OK = (0, 255, 0)
    COLOR_WARN = (0, 255, 255)
    COLOR_BAD = (0, 0, 255)

    # --------------------------------------------------------
    # 基礎數學
    # --------------------------------------------------------

    def calculate_angle(self, a, b, c):
        """
        計算：

            A
             \
              B
             /
            C

        B 是角度頂點。
        """

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(
            ba,
            bc
        ) / (
            np.linalg.norm(ba)
            * np.linalg.norm(bc)
        )

        cosine_angle = np.clip(
            cosine_angle,
            -1.0,
            1.0
        )

        angle = np.arccos(
            cosine_angle
        )

        return np.degrees(angle)

    # ========================================================
    # Front Pipeline
    # ========================================================

    def analyze_front(self, points):
        """
        Front Analysis Pipeline

            Landmarks
                ↓
            Shoulder
                ↓
            Hip
                ↓
            Legs
                ↓
            Arms
                ↓
            Result
        """

        result = {}

        # ----------------------------------------------------
        # 1. Shoulder
        # ----------------------------------------------------

        l_shoulder = points[11]
        r_shoulder = points[12]

        shoulder_slope = math.degrees(
            math.atan2(
                r_shoulder[1] - l_shoulder[1],
                r_shoulder[0] - l_shoulder[0]
            )
        )

        shoulder_diff = abs(
            shoulder_slope
            if abs(shoulder_slope) < 90
            else 180 - abs(shoulder_slope)
        )

        shoulder_status = "Shoulders OK"

        if shoulder_diff > 3:
            shoulder_status = "Uneven Shoulders"

        result["shoulder"] = {
            "diff": shoulder_diff,
            "status": shoulder_status,
            "points": [
                l_shoulder,
                r_shoulder
            ]
        }

        # ----------------------------------------------------
        # 2. Hip
        # ----------------------------------------------------

        l_hip = points[23]
        r_hip = points[24]

        hip_slope = math.degrees(
            math.atan2(
                r_hip[1] - l_hip[1],
                r_hip[0] - l_hip[0]
            )
        )

        hip_diff = abs(
            hip_slope
            if abs(hip_slope) < 90
            else 180 - abs(hip_slope)
        )

        hip_status = "Pelvis OK"

        if hip_diff > 3:

            if l_hip[1] < r_hip[1]:
                hip_status = "R-Leg Short (Tilt)"
            else:
                hip_status = "L-Leg Short (Tilt)"

        result["hip"] = {
            "diff": hip_diff,
            "status": hip_status,
            "points": [
                l_hip,
                r_hip
            ]
        }

        # ----------------------------------------------------
        # 3. Legs
        # ----------------------------------------------------

        angle_l_leg = self.calculate_angle(
            points[23],
            points[25],
            points[27]
        )

        angle_r_leg = self.calculate_angle(
            points[24],
            points[26],
            points[28]
        )

        avg_leg_angle = (
            angle_l_leg
            + angle_r_leg
        ) / 2

        leg_status = "Legs Alignment OK"

        if avg_leg_angle < 170:
            leg_status = "X-Type Legs (Valgus)"

        elif avg_leg_angle > 185:
            leg_status = "O-Type Legs (Varus)"

        result["legs"] = {
            "left_angle": angle_l_leg,
            "right_angle": angle_r_leg,
            "average_angle": avg_leg_angle,
            "status": leg_status
        }

        # ----------------------------------------------------
        # 4. Arms
        # ----------------------------------------------------

        angle_l_arm = self.calculate_angle(
            points[11],
            points[13],
            points[15]
        )

        angle_r_arm = self.calculate_angle(
            points[12],
            points[14],
            points[16]
        )

        arm_status = "Arm Alignment OK"

        if (
            angle_l_arm < 160
            or angle_r_arm < 160
        ):
            arm_status = "Cubitus Valgus (Elbow)"

        result["arms"] = {
            "left_angle": angle_l_arm,
            "right_angle": angle_r_arm,
            "status": arm_status
        }

        return result

    # ========================================================
    # Side Pipeline
    # ========================================================

    def analyze_side(self, points, landmarks):
        """
        Side Analysis Pipeline

            Landmarks
                ↓
            View Detection
                ↓
            Neck
                ↓
            Shoulder
                ↓
            Knee
                ↓
            Result
        """

        result = {}

        # ----------------------------------------------------
        # 1. 判斷哪一側
        # ----------------------------------------------------

        left_visibility = landmarks[11].visibility
        right_visibility = landmarks[12].visibility

        if left_visibility > right_visibility:

            side = "Left"

            ear = 7
            shoulder = 11
            hip = 23
            knee = 25
            ankle = 27

        else:

            side = "Right"

            ear = 8
            shoulder = 12
            hip = 24
            knee = 26
            ankle = 28

        result["view"] = side

        # ----------------------------------------------------
        # 2. Forward Head
        # ----------------------------------------------------

        pt_ear = points[ear]
        pt_shoulder = points[shoulder]

        vertical_point = [
            pt_shoulder[0],
            pt_shoulder[1] - 100
        ]

        neck_angle = self.calculate_angle(
            pt_ear,
            pt_shoulder,
            vertical_point
        )

        neck_status = "Neck Position OK"

        if neck_angle > 30:
            neck_status = "Forward Head (Turtle Neck)"

        elif neck_angle > 15:
            neck_status = "Slight Forward Head"

        result["neck"] = {
            "angle": neck_angle,
            "status": neck_status,
            "points": [
                pt_ear,
                pt_shoulder,
                vertical_point
            ]
        }

        # ----------------------------------------------------
        # 3. Rounded Shoulder
        # ----------------------------------------------------

        pt_hip = points[hip]

        shoulder_hip_offset = (
            pt_shoulder[0]
            - pt_hip[0]
        )

        is_rounded = False

        if (
            side == "Left"
            and shoulder_hip_offset < -20
        ):
            is_rounded = True

        if (
            side == "Right"
            and shoulder_hip_offset > 20
        ):
            is_rounded = True

        rs_status = "Shoulder Position OK"

        if is_rounded:
            rs_status = "Rounded Shoulders"

        result["rounded_shoulder"] = {
            "offset": shoulder_hip_offset,
            "status": rs_status,
            "points": [
                pt_shoulder,
                pt_hip
            ]
        }

        # ----------------------------------------------------
        # 4. Knee
        # ----------------------------------------------------

        pt_knee = points[knee]
        pt_ankle = points[ankle]

        knee_angle = self.calculate_angle(
            pt_hip,
            pt_knee,
            pt_ankle
        )

        knee_status = "Knee Normal"

        if knee_angle < 165:
            knee_status = "Knee Flexion (Bent)"

        result["knee"] = {
            "angle": knee_angle,
            "status": knee_status,
            "points": [
                pt_hip,
                pt_knee,
                pt_ankle
            ]
        }

        return result


# ============================================================
# 7. Result → CSV
# ============================================================

class ResultFormatter:

    def to_csv_record(self, result, timestamp):
        """
        將 Analysis Result
        轉成 CSV 可以使用的格式。

        注意：
        Analysis 不負責 CSV。
        這裡才負責資料格式轉換。
        """

        record = {
            column: None
            for column in COLUMNS
        }

        record["timestamp"] = timestamp

        # ----------------------------------------------------
        # Front Result
        # ----------------------------------------------------

        if "shoulder" in result:

            record["shldr_diff"] = round(
                result["shoulder"]["diff"],
                2
            )

            record["hip_diff"] = round(
                result["hip"]["diff"],
                2
            )

            record["angle_l_leg"] = (
                result["legs"]["left_angle"]
            )

            record["angle_r_leg"] = (
                result["legs"]["right_angle"]
            )

            record["angle_l_arm"] = (
                result["arms"]["left_angle"]
            )

            record["angle_r_arm"] = (
                result["arms"]["right_angle"]
            )

        # ----------------------------------------------------
        # Side Result
        # ----------------------------------------------------

        if "neck" in result:

            record["neck_angle"] = round(
                result["neck"]["angle"],
                2
            )

            record["neck_mode"] = (
                result["neck"]["status"]
            )

        if "knee" in result:

            record["knee_angle"] = round(
                result["knee"]["angle"],
                2
            )

            record["knee_mode"] = (
                0.0
                if result["knee"]["angle"] < 165
                else 1.0
            )

        if "rounded_shoulder" in result:

            record["shldr_diff"] = round(
                result["rounded_shoulder"]["offset"],
                2
            )

        return record


# ============================================================
# 8. Visualization Pipeline
# ============================================================

class Visualizer:

    COLOR_OK = (0, 255, 0)
    COLOR_WARN = (0, 255, 255)
    COLOR_BAD = (0, 0, 255)

    def draw_line(
        self,
        image,
        p1,
        p2,
        color,
        thickness=2
    ):

        cv2.line(
            image,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color,
            thickness
        )

    def draw_points(
        self,
        image,
        points
    ):

        for point in points:

            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                5,
                (255, 255, 255),
                -1
            )

    def draw_text(
        self,
        image,
        text,
        position,
        color
    ):

        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ========================================================
    # Front Visualization
    # ========================================================

    def draw_front(self, image, result):

        height, width, _ = image.shape

        x = 20
        y = 30
        line_height = 35

        # ----------------------------------------------------
        # Shoulder
        # ----------------------------------------------------

        shoulder = result["shoulder"]

        color = (
            self.COLOR_BAD
            if shoulder["diff"] > 3
            else self.COLOR_OK
        )

        self.draw_line(
            image,
            shoulder["points"][0],
            shoulder["points"][1],
            color
        )

        self.draw_text(
            image,
            f"Shoulder Tilt: {shoulder['diff']:.1f}",
            (x, y),
            color
        )

        y += line_height

        self.draw_text(
            image,
            f"Status: {shoulder['status']}",
            (x, y),
            color
        )

        y += line_height + 10

        # ----------------------------------------------------
        # Hip
        # ----------------------------------------------------

        hip = result["hip"]

        color = (
            self.COLOR_BAD
            if hip["diff"] > 3
            else self.COLOR_OK
        )

        self.draw_line(
            image,
            hip["points"][0],
            hip["points"][1],
            color
        )

        self.draw_text(
            image,
            f"Hip Tilt: {hip['diff']:.1f}",
            (x, y),
            color
        )

        y += line_height

        self.draw_text(
            image,
            f"Status: {hip['status']}",
            (x, y),
            color
        )

        y += line_height + 10

        # ----------------------------------------------------
        # Legs
        # ----------------------------------------------------

        legs = result["legs"]

        if legs["average_angle"] < 170:
            color = self.COLOR_BAD

        elif legs["average_angle"] > 185:
            color = self.COLOR_WARN

        else:
            color = self.COLOR_OK

        self.draw_line(
            image,
            result["legs"].get(
                "left_points",
                [0, 0]
            ),
            result["legs"].get(
                "right_points",
                [0, 0]
            ),
            color
        )

        self.draw_text(
            image,
            f"Avg Leg Angle: {legs['average_angle']:.1f}",
            (x, y),
            color
        )

        y += line_height

        self.draw_text(
            image,
            f"Status: {legs['status']}",
            (x, y),
            color
        )

        y += line_height + 10

        # ----------------------------------------------------
        # Arms
        # ----------------------------------------------------

        arms = result["arms"]

        color = (
            self.COLOR_WARN
            if (
                arms["left_angle"] < 160
                or arms["right_angle"] < 160
            )
            else self.COLOR_OK
        )

        self.draw_text(
            image,
            f"Status: {arms['status']}",
            (x, y),
            color
        )

        return image

    # ========================================================
    # Side Visualization
    # ========================================================

    def draw_side(self, image, result):

        x = 20
        y = 40
        line_height = 35

        # ----------------------------------------------------
        # View
        # ----------------------------------------------------

        view = result["view"]

        self.draw_text(
            image,
            f"{view} Side View",
            (x, y),
            (200, 200, 200)
        )

        y += 45

        # ----------------------------------------------------
        # Neck
        # ----------------------------------------------------

        neck = result["neck"]

        if neck["angle"] > 30:
            color = self.COLOR_BAD

        elif neck["angle"] > 15:
            color = self.COLOR_WARN

        else:
            color = self.COLOR_OK

        self.draw_line(
            image,
            neck["points"][0],
            neck["points"][1],
            color,
            4
        )

        self.draw_line(
            image,
            neck["points"][1],
            neck["points"][2],
            (255, 255, 0)
        )

        self.draw_text(
            image,
            f"Neck Angle: {int(neck['angle'])} deg",
            (x, y),
            color
        )

        y += line_height

        self.draw_text(
            image,
            f"Status: {neck['status']}",
            (x, y),
            color
        )

        y += line_height + 10

        # ----------------------------------------------------
        # Rounded Shoulder
        # ----------------------------------------------------

        shoulder = result["rounded_shoulder"]

        color = (
            self.COLOR_WARN
            if shoulder["status"] == "Rounded Shoulders"
            else self.COLOR_OK
        )

        self.draw_line(
            image,
            shoulder["points"][0],
            shoulder["points"][1],
            color
        )

        self.draw_text(
            image,
            "Shoulder-Hip Align",
            (x, y),
            color
        )

        y += line_height

        self.draw_text(
            image,
            f"Status: {shoulder['status']}",
            (x, y),
            color
        )

        y += line_height + 10

        # ----------------------------------------------------
        # Knee
        # ----------------------------------------------------

        knee = result["knee"]

        color = (
            self.COLOR_WARN
            if knee["angle"] < 165
            else self.COLOR_OK
        )

        points = knee["points"]

        self.draw_line(
            image,
            points[0],
            points[1],
            color
        )

        self.draw_line(
            image,
            points[1],
            points[2],
            color
        )

        self.draw_points(
            image,
            points
        )

        self.draw_text(
            image,
            f"Knee Angle: {int(knee['angle'])} deg",
            (x, y),
            color
        )

        y += line_height

        self.draw_text(
            image,
            f"Status: {knee['status']}",
            (x, y),
            color
        )

        return image


# ============================================================
# 9. Capture Pipeline
# ============================================================

def countdown():

    for i in range(3, 0, -1):

        print(f"Capturing in {i}...")

        cv2.waitKey(1000)


# ============================================================
# 10. Main Pipeline
# ============================================================

def main():

    # --------------------------------------------------------
    # 建立 Pipeline Components
    # --------------------------------------------------------

    camera = Camera()
    detector = PoseDetector()
    extractor = LandmarkExtractor()
    analyzer = PostureAnalyzer()
    formatter = ResultFormatter()
    visualizer = Visualizer()
    storage = DataStorage(DATA_FILE)

    print("=== Advanced Posture System ===")
    print("Press 'f' for Front View")
    print("Press 's' for Side View")
    print("Press 'q' to Quit")


    # ========================================================
    # Main Loop
    # ========================================================

    while True:

        # ====================================================
        # Pipeline 1
        # Camera → Frame
        # ====================================================

        frame = camera.read()

        if frame is None:
            break


        # ====================================================
        # Pipeline 2
        # Frame → Preview
        # ====================================================

        display = frame.copy()

        cv2.putText(
            display,
            "Press 'f' (Front) / 's' (Side) / 'q'",
            (20, 700),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Preview",
            display
        )


        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # Pipeline 3
        # User → Capture
        # ====================================================

        if key == ord("f") or key == ord("s"):

            mode = (
                "Front"
                if key == ord("f")
                else "Side"
            )

            print(f"\n=== {mode} Capture ===")

            countdown()


            # ------------------------------------------------
            # 取得真正分析的 Frame
            # ------------------------------------------------

            clean_frame = camera.read()

            if clean_frame is None:
                break


            # =================================================
            # Pipeline 4
            # Frame → Landmarks
            # =================================================

            landmarks = detector.detect(
                clean_frame
            )

            if landmarks is None:

                print(
                    "No pose detected."
                )

                continue


            # =================================================
            # Pipeline 5
            # Landmarks → Points
            # =================================================

            points = extractor.extract(
                landmarks,
                clean_frame
            )


            # =================================================
            # Pipeline 6
            # Points → Analysis
            # =================================================

            if mode == "Front":

                result = analyzer.analyze_front(
                    points
                )

            else:

                result = analyzer.analyze_side(
                    points,
                    landmarks
                )


            # =================================================
            # Pipeline 7
            # Analysis → CSV Record
            # =================================================

            timestamp = len(storage.data)

            record = formatter.to_csv_record(
                result,
                timestamp
            )

            storage.save(record)


            # =================================================
            # Pipeline 8
            # Analysis → Visualization
            # =================================================

            result_image = clean_frame.copy()

            if mode == "Front":

                result_image = visualizer.draw_front(
                    result_image,
                    result
                )

                filename = "result_front_adv.jpg"

            else:

                result_image = visualizer.draw_side(
                    result_image,
                    result
                )

                filename = "result_side_adv.jpg"


            # =================================================
            # Pipeline 9
            # Result → Output
            # =================================================

            cv2.imshow(
                "Result",
                result_image
            )

            cv2.imwrite(
                filename,
                result_image
            )

            print(
                f"Saved {filename}"
            )

            cv2.waitKey(0)

            cv2.destroyWindow(
                "Result"
            )


        # ====================================================
        # Quit
        # ====================================================

        elif key == ord("q"):

            storage.write()

            break


    # ========================================================
    # Cleanup
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()