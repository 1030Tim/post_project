import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image, ImageDraw, ImageFont

# --- 1. 統一 CSV 欄位定義 ---
COLUMNS = ["時間", "視角", "頸部角度", "膝蓋角度", "肩膀偏移", "骨盆偏移", 
           "左腿角度", "右腿角度", "左臂角度", "右臂角度", "異常症狀"]
CSV_FILE = "data.csv"

# --- 2. 處方箋資料庫 (請根據需求修改內容與連結) ---
PRESCRIPTIONS = {
    "X型腿": {
        "advice": (
            "<b>【問題說明】</b>：膝蓋向內（X型），會影響走路與關節受力。<br>"
            "<b>【改善目標】</b>：讓膝關節回到正確排列，減少關節壓力。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 臀中肌強化（防止膝內夾）<br>"
            "2. 大腿外側訓練<br>"
            "<b>【日常注意】</b>：避免內八或外八走路；深蹲時注意膝蓋方向；穿著合適鞋子。"
        ),
        "video": "https://youtu.be/VVaTglfUBWI"
    },
    "O型腿": {
        "advice": (
            "<b>【問題說明】</b>：膝蓋向外（O型），會影響走路與關節受力。<br>"
            "<b>【改善目標】</b>：讓膝關節回到正確排列，減少關節壓力。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 臀中肌強化（防止膝內夾）<br>"
            "2. 大腿外側訓練<br>"
            "<b>【日常注意】</b>：避免內八或外八走路；深蹲時注意膝蓋方向；穿著合適鞋子。"
        ),
        "video": "https://youtu.be/VVaTglfUBWI"
    },
    "嚴重烏龜頸": {
        "advice": (
            "<b>【問題說明】</b>：頭部過度向前，常見於長時間使用手機或電腦。<br>"
            "<b>【改善目標】</b>：讓頭部回到身體正上方，減少頸椎壓力。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 下巴內收運動（Chin tuck）<br>"
            "2. 頸後肌群強化<br>"
            "3. 胸肌伸展<br>"
            "<b>【日常注意】</b>：手機抬高至眼睛高度；螢幕與眼睛保持水平；避免長時間低頭"
        ),
        "video": "https://youtu.be/61I1lubBr9s"
    },
    "輕微烏龜頸": {
        "advice": (
            "<b>【問題說明】</b>：頭部過度向前，常見於長時間使用手機或電腦。<br>"
            "<b>【改善目標】</b>：讓頭部回到身體正上方，減少頸椎壓力。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 下巴內收運動（Chin tuck）<br>"
            "2. 頸後肌群強化<br>"
            "3. 胸肌伸展<br>"
            "<b>【日常注意】</b>：手機抬高至眼睛高度；螢幕與眼睛保持水平；避免長時間低頭"
        ),
        "video": "https://youtu.be/61I1lubBr9s"
    },
    "圓肩": {
        "advice": (
            "<b>【問題說明】</b>：肩膀向前內縮，常伴隨駝背與頭部前傾，多因長時間使用電腦、滑手機或胸肌過緊造成。<br>"
            "<b>【改善目標】</b>：打開胸口、讓肩膀回到身體側邊，恢復正常排列。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 胸肌伸展（門框伸展）<br>"
            "2. 肩胛後收訓練（夾背動作）<br>"
            "3. 划船動作（彈力帶或啞鈴）<br>"
            "4. 背部肌群強化（菱形肌、中下斜方肌）<br>"
            "<b>【日常注意】</b>：避免駝背坐姿;電腦螢幕高度與視線平行;手機使用時抬高至胸口以上;定時活動肩膀（每30–60分鐘）"
        ),
        "video": "https://youtu.be/Aql3FVyaoUc"
    },
    "高低肩": {
        "advice": (
            "<b>【問題說明】</b>：左右肩膀高度不一致，可能與肌肉不平衡、單側用力或長期姿勢習慣有關。<br>"
            "<b>【改善目標】</b>：恢復肩膀左右對稱，改善肌肉不平衡。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 肩頸伸展（特別是較高的一側）<br>"
            "2. 單側肩胛穩定訓練（如側平舉、彈力帶拉伸）<br>"
            "3. 上斜方肌放鬆（使用滾筒或按摩球）<br>"
            "<b>【日常注意】</b>：避免長期單肩背包；使用滑鼠或手機時避免固定單側用力；坐姿保持雙肩放鬆水平。"
        ),
        "video": "https://youtu.be/dWxXA70ZbDo"
    },
    "骨盆傾斜": {
        "advice": (
            "<b>【問題說明】</b>：骨盆向前傾斜，常見於久坐族，會導致腰椎壓力增加與小腹突出。<br>"
            "<b>【改善目標】</b>：恢復骨盆中立位置，減少腰椎負擔。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 臀肌強化（深蹲、橋式）<br>"
            "2. 腹核心訓練（平板撐）<br>"
            "3. 髂腰肌伸展（髖前側）<br>"
            "<b>【日常注意】</b>：避免長時間久坐;站姿避免「翹屁股」;坐姿保持骨盆直立。"
        ),
        "video": "https://youtu.be/iGae8R7jB60"
    },
    "反曲膝": {
        "advice": (
            "<b>【問題說明】</b>：站立時膝蓋過度向後伸直（鎖死），會增加膝關節壓力，長期可能導致關節不穩或疼痛。<br>"
            "<b>【改善目標】</b>：讓膝蓋維持「微彎」狀態，提升關節穩定與肌肉控制。<br>"
            "<b>【建議訓練】</b>：<br>"
            "1. 大腿後側（腿後肌）強化（如腿彎舉）<br>"
            "2. 股四頭肌控制訓練<br>"
            "3. 平衡與穩定訓練（單腳站）<br>"
            "4. 核心穩定訓練（避免代償）<br>"
            "<b>【日常注意】</b>：避免站立時膝蓋完全打直;長時間站立時微彎膝蓋;運動時注意膝關節控制;避免「鎖膝」習慣。"
        ),
        "video": "https://youtu.be/deXX97mhd8E"
    }
}

# --- 3. 郵件發送功能 ---
def send_prescription_email(target_email, detected_abnormals):
    """ 根據偵測到的異常症狀清單，寄送對應的處方箋 """
    if not detected_abnormals:
        return
    
    # --- 重要：請在此設定您的 Gmail 帳號與應用程式密碼 ---
    sender_email = "jun20061030@gmail.com" 
    app_password = "rwgi tluc pyce zumi" # 請使用 Google 生成的 16 位「應用程式密碼」

    msg = MIMEMultipart()
    msg['Subject'] = "【體態檢測報告】您的專屬改善處方箋"
    msg['From'] = f"體態分析系統 <{sender_email}>"
    msg['To'] = target_email

    # 組合 HTML 內容
    html_content = f"<h3>您好，這是根據您剛剛檢測結果提供的改善建議：</h3>"
    html_content += "<table border='1' cellpadding='10' style='border-collapse: collapse;'>"
    html_content += "<tr style='background-color: #f2f2f2;'><th>偵測症狀</th><th>改善建議</th><th>處方箋影片</th></tr>"

    found_any = False
    for symptom_key in PRESCRIPTIONS.keys():
        # 檢查偵測到的症狀中是否包含關鍵字 (例如 "骨盆傾斜(左腳短)" 包含 "骨盆傾斜")
        for detected in detected_abnormals:
            if symptom_key in detected:
                info = PRESCRIPTIONS[symptom_key]
                html_content += f"<tr><td><b>{detected}</b></td><td>{info['advice']}</td>"
                html_content += f"<td><a href='{info['video']}'>點我觀看教學</a></td></tr>"
                found_any = True
                break
    
    html_content += "</table><p>持續練習可以改善體態，祝您健康！</p>"

    if not found_any:
        return

    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print(f"✅ 處方箋已成功寄送到: {target_email}")
    except Exception as e:
        print(f"❌ 郵件寄送失敗: {e}")

class AdvancedPostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        self.COLOR_OK = (0, 255, 0)
        self.COLOR_WARN = (0, 255, 255)
        self.COLOR_BAD = (0, 0, 255)
        
        # Mac 字型設定
        self.font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
        if not os.path.exists(self.font_path):
            self.font_path = "/System/Library/Fonts/PingFang.ttc"

    def draw_text_ch(self, image, text, pos, color, size=22):
        """ 支援中文顯示的文字繪製 """
        if not os.path.exists(self.font_path):
            cv2.putText(image, "Font Error", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return image
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.font_path, size)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def draw_analysis_ch(self, image, points, angle, label, color, pos):
        """ 繪製連線、關節點與角度標籤 """
        cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color, 3)
        cv2.line(image, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])), color, 3)
        for p in points:
            cv2.circle(image, (int(p[0]), int(p[1])), 6, (255, 255, 255), -1)
            cv2.circle(image, (int(p[0]), int(p[1])), 3, color, -1)
        image = self.draw_text_ch(image, f"{label}: {int(angle)}°", pos, color, 18)
        return image

    def analyze_front(self, image):
        h, w, _ = image.shape
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        data = {col: "" for col in COLUMNS}
        if not results.pose_landmarks: return image, data

        lm = results.pose_landmarks.landmark
        def get_pt(idx): return [lm[idx].x * w, lm[idx].y * h]

        data.update({"時間": time.strftime("%Y-%m-%d %H:%M:%S"), "視角": "正面"})
        
        display_status = []
        csv_abnormal = []

        # 1. 高低肩
        l_sh, r_sh = get_pt(11), get_pt(12)
        sh_slope = math.degrees(math.atan2(r_sh[1]-l_sh[1], r_sh[0]-l_sh[0]))
        sh_diff = abs(sh_slope if abs(sh_slope) < 90 else 180 - abs(sh_slope))
        data["肩膀偏移"] = round(sh_diff, 2)
        if sh_diff > 3:
            sh_status, sh_col = "高低肩 (不正常)", self.COLOR_BAD
            csv_abnormal.append("高低肩")
        else:
            sh_status, sh_col = "肩膀 (正常)", self.COLOR_OK
        display_status.append((f"肩膀: {sh_status}", sh_col))
        cv2.line(image, (int(l_sh[0]), int(l_sh[1])), (int(r_sh[0]), int(r_sh[1])), sh_col, 3)

        # 2. 骨盆
        l_h, r_h = get_pt(23), get_pt(24)
        hp_slope = math.degrees(math.atan2(r_h[1]-l_h[1], r_h[0]-l_h[0]))
        hp_diff = abs(hp_slope if abs(hp_slope) < 90 else 180 - abs(hp_slope))
        data["骨盆偏移"] = round(hp_diff, 2)
        if hp_diff > 3:
            side_text = "右腳短" if l_h[1] < r_h[1] else "左腳短"
            hp_status, hp_col = f"骨盆傾斜-{side_text}", self.COLOR_BAD
            csv_abnormal.append(f"骨盆傾斜({side_text})")
        else:
            hp_status, hp_col = "骨盆 (正常)", self.COLOR_OK
        display_status.append((f"骨盆: {hp_status}", hp_col))
        cv2.line(image, (int(l_h[0]), int(l_h[1])), (int(r_h[0]), int(r_h[1])), hp_col, 3)

        # 3. 腿部
        ang_l_leg = self.calculate_angle(get_pt(23), get_pt(25), get_pt(27))
        ang_r_leg = self.calculate_angle(get_pt(24), get_pt(26), get_pt(28))
        data["左腿角度"], data["右腿角度"] = round(ang_l_leg, 1), round(ang_r_leg, 1)
        avg_leg = (ang_l_leg + ang_r_leg) / 2
        if avg_leg < 170:
            leg_status, leg_col = "X型腿 (不正常)", self.COLOR_BAD
            csv_abnormal.append("X型腿")
        elif avg_leg > 185:
            leg_status, leg_col = "O型腿 (警告)", self.COLOR_WARN
            csv_abnormal.append("O型腿")
        else:
            leg_status, leg_col = "腿部 (正常)", self.COLOR_OK
        display_status.append((f"腿部: {leg_status}", leg_col))
        
        image = self.draw_analysis_ch(image, [get_pt(23), get_pt(25), get_pt(27)], ang_l_leg, "左腿", leg_col, (20, h-60))
        image = self.draw_analysis_ch(image, [get_pt(24), get_pt(26), get_pt(28)], ang_r_leg, "右腿", leg_col, (w-150, h-60))

        for i, (text, col) in enumerate(display_status):
            image = self.draw_text_ch(image, text, (w-280, 30 + i*35), col, 20)

        data["異常症狀"] = ", ".join(csv_abnormal) if csv_abnormal else "無"
        return image, data

    def analyze_side(self, image):
        h, w, _ = image.shape
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        data = {col: "" for col in COLUMNS}
        if not results.pose_landmarks: return image, data

        lm = results.pose_landmarks.landmark
        def get_pt(idx): return [lm[idx].x * w, lm[idx].y * h]

        side = "左側" if lm[11].visibility > lm[12].visibility else "右側"
        ear, sh, hp, kn, ak = (7, 11, 23, 25, 27) if side == "左側" else (8, 12, 24, 26, 28)
        p_ear, p_sh, p_hp, p_kn, p_ak = get_pt(ear), get_pt(sh), get_pt(hp), get_pt(kn), get_pt(ak)

        data.update({"時間": time.strftime("%Y-%m-%d %H:%M:%S"), "視角": f"側面({side})"})
        
        display_status = []
        csv_abnormal = []

        # 1. 烏龜頸
        neck_ang = self.calculate_angle(p_ear, p_sh, [p_sh[0], p_sh[1]-100])
        data["頸部角度"] = round(neck_ang, 1)
        if neck_ang > 30:
            n_status, n_col = "嚴重烏龜頸", self.COLOR_BAD
            csv_abnormal.append("嚴重烏龜頸")
        elif neck_ang > 15:
            n_status, n_col = "輕微烏龜頸", self.COLOR_WARN
            csv_abnormal.append("輕微烏龜頸")
        else:
            n_status, n_col = "頸部 (正常)", self.COLOR_OK
        display_status.append((f"頸部: {n_status}", n_col))
        
        cv2.line(image, (int(p_sh[0]), int(p_sh[1])), (int(p_ear[0]), int(p_ear[1])), n_col, 4)
        cv2.line(image, (int(p_sh[0]), int(p_sh[1])), (int(p_sh[0]), int(p_sh[1]-100)), (255, 255, 0), 2)

        # 2. 圓肩
        offset = p_sh[0] - p_hp[0]
        data["肩膀偏移"] = round(offset, 2)
        is_rounded = (side == "左側" and offset < -20) or (side == "右側" and offset > 20)
        if is_rounded:
            rs_status, rs_col = "圓肩 (警告)", self.COLOR_WARN
            csv_abnormal.append("圓肩")
        else:
            rs_status, rs_col = "肩部 (正常)", self.COLOR_OK
        display_status.append((f"肩膀: {rs_status}", rs_col))
        cv2.line(image, (int(p_sh[0]), int(p_sh[1])), (int(p_hp[0]), int(p_hp[1])), rs_col, 2)

        # 3. 膝蓋
        knee_ang = self.calculate_angle(p_hp, p_kn, p_ak)
        data["膝蓋角度"] = round(knee_ang, 1)
        if knee_ang < 165:
            k_status, k_col = "反曲膝", self.COLOR_WARN
            csv_abnormal.append("反曲膝")
        else:
            k_status, k_col = "膝蓋 (正常)", self.COLOR_OK
        display_status.append((f"膝蓋: {k_status}", k_col))
        image = self.draw_analysis_ch(image, [p_hp, p_kn, p_ak], knee_ang, "膝蓋", k_col, (20, h-50))

        for i, (text, col) in enumerate(display_status):
            image = self.draw_text_ch(image, text, (w-280, 30 + i*35), col, 20)

        data["異常症狀"] = ", ".join(csv_abnormal) if csv_abnormal else "無"
        return image, data

def main():
    # A. 詢問使用者電子郵件
    user_email = input("請輸入您的 Email 以便接收處方箋: ").strip()

    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if list(df.columns) != COLUMNS:
                df = pd.DataFrame(columns=COLUMNS)
        except:
            df = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    cap = cv2.VideoCapture(0)
    analyzer = AdvancedPostureAnalyzer()
    workflow = [("正面", 1), ("正面", 2), ("側面", 1), ("側面", 2)]
    session_results = []
    all_session_abnormals = [] # 用來收集這一次檢測所有的症狀

    print("=== 系統啟動 ===")

    while len(session_results) < 4:
        v_type, count = workflow[len(session_results)]
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        display = analyzer.draw_text_ch(display, f"任務: {v_type} ({count}/2)", (20, 420), (255, 255, 0), 25)
        display = analyzer.draw_text_ch(display, "按 [空白鍵] 拍照分析", (20, 450), (255, 255, 255), 20)
        cv2.imshow("Mac Posture Camera", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            ret, clean = cap.read()
            res_img, data = analyzer.analyze_front(clean) if v_type == "正面" else analyzer.analyze_side(clean)
            
            session_results.append(data)
            # 收集異常症狀供後續寄信
            if data["異常症狀"] != "無":
                all_session_abnormals.extend(data["異常症狀"].split(", "))
            
            cv2.imshow("Result", res_img)
            cv2.waitKey(2000)
            cv2.destroyWindow("Result")
        elif key == ord('q'): break

    if session_results:
        # 1. 存檔到 CSV
        new_df = pd.DataFrame(session_results)
        new_df = new_df[COLUMNS]
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"存檔完成！共儲存 {len(df)} 筆。")

        # 2. 寄送處方箋 Email
        if user_email:
            # 移除重複的症狀
            unique_abnormals = list(set(all_session_abnormals))
            print(f"準備為以下症狀寄送處方箋: {unique_abnormals}")
            send_prescription_email(user_email, unique_abnormals)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()