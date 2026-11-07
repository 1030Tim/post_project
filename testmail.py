"""Posture Guard: a Tkinter posture-screening and tracking application."""
from __future__ import annotations

import base64
import csv
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "posture_data"
APP_VERSION, PROTOCOL_VERSION = "2.0.0", "2026-11-standardized-1"
WORKFLOW = [(view, n) for view in ("正面", "左側", "右側") for n in range(1, 4)]
SESSION_HEADERS = ["session_id", "participant_id", "measured_at", "app_version", "protocol_version", "view", "capture_number", "image_quality", "quality_note", "shoulder_tilt_deg", "pelvic_tilt_deg", "head_neck_proxy_deg", "knee_angle_deg", "left_leg_angle_deg", "right_leg_angle_deg", "screening_result"]
SUMMARY_HEADERS = ["session_id", "participant_id", "measured_at", "app_version", "protocol_version", "front_shoulder_tilt_median_deg", "front_pelvic_tilt_median_deg", "side_head_neck_proxy_median_deg", "side_knee_median_deg", "quality_pass_rate", "screening_result", "review_status"]
REVIEW_HEADERS = ["session_id", "reviewed_at", "reviewer_id", "clinical_reference", "review_result", "notes"]
AUDIT_HEADERS = ["timestamp", "event", "session_id", "details"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


class CsvStore:
    """Operational, summary, review, and audit data are deliberately separate."""
    files = {"sessions": ("sessions.csv", SESSION_HEADERS), "summaries": ("summaries.csv", SUMMARY_HEADERS), "reviews": ("reviews.csv", REVIEW_HEADERS), "audit": ("audit_log.csv", AUDIT_HEADERS)}

    def __init__(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        for name, headers in self.files.values():
            path = DATA_DIR / name
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8-sig") as f:
                    csv.DictWriter(f, fieldnames=headers).writeheader()

    def append(self, kind: str, rows: list[dict[str, str]]) -> None:
        name, headers = self.files[kind]
        with (DATA_DIR / name).open("a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=headers, extrasaction="ignore").writerows(rows)

    def audit(self, event: str, session_id: str = "", details: str = "") -> None:
        self.append("audit", [{"timestamp": now(), "event": event, "session_id": session_id, "details": details}])

    def read(self, kind: str) -> list[dict[str, str]]:
        name, _ = self.files[kind]
        with (DATA_DIR / name).open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def mark_reviewed(self, session_id: str) -> None:
        """Mark a completed session without altering its original measurements."""
        rows = self.read("summaries")
        for row in rows:
            if row["session_id"] == session_id:
                row["review_status"] = "已專業覆核"
        name, headers = self.files["summaries"]
        with (DATA_DIR / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


@dataclass
class Analysis:
    image: np.ndarray
    row: dict[str, str]


class PostureAnalyzer:
    """2D landmark measurements; screening prompts, not a clinical diagnosis."""
    def __init__(self) -> None:
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "需要 MediaPipe 0.10.x。請執行：python3 -m pip install -r requirements.txt --upgrade"
            )
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.60)

    @staticmethod
    def point(landmarks, index: int, width: int, height: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array([landmark.x * width, landmark.y * height], dtype=float)

    @staticmethod
    def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        x, y = a - b, c - b
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return 0.0 if denom < 1e-8 else float(np.degrees(np.arccos(np.clip(np.dot(x, y) / denom, -1, 1))))

    @staticmethod
    def line(image: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int]) -> None:
        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), color, 3, cv2.LINE_AA)
        cv2.circle(image, tuple(a.astype(int)), 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, tuple(b.astype(int)), 5, color, -1, cv2.LINE_AA)

    def analyze(self, frame: np.ndarray, view: str, session_id: str, participant: str, capture_number: int) -> Analysis:
        image = frame.copy()
        height, width = image.shape[:2]
        row = {key: "" for key in SESSION_HEADERS}
        row.update({"session_id": session_id, "participant_id": participant, "measured_at": now(), "app_version": APP_VERSION, "protocol_version": PROTOCOL_VERSION, "view": view, "capture_number": str(capture_number), "image_quality": "不合格"})
        result = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            row.update({"quality_note": "未偵測到人體", "screening_result": "不納入量測"})
            return Analysis(image, row)
        lm = result.pose_landmarks.landmark
        needed = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        quality = min(lm[i].visibility for i in needed)
        points = [self.point(lm, i, width, height) for i in needed]
        if quality < 0.55 or not all(20 < p[0] < width - 20 and 20 < p[1] < height - 20 for p in points):
            row.update({"quality_note": "關鍵點可信度不足或身體未完整入鏡", "screening_result": "不納入量測"})
            return Analysis(image, row)
        row.update({"image_quality": "合格", "quality_note": f"最低關鍵點可信度 {quality:.2f}"})
        p = lambda i: self.point(lm, i, width, height)
        green, amber, red = (50, 180, 80), (0, 180, 230), (40, 60, 230)
        alerts: list[str] = []
        if view == "正面":
            ls, rs, lh, rh = p(11), p(12), p(23), p(24)
            shoulder = abs(math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])))
            pelvic = abs(math.degrees(math.atan2(rh[1] - lh[1], rh[0] - lh[0])))
            shoulder, pelvic = min(shoulder, abs(180 - shoulder)), min(pelvic, abs(180 - pelvic))
            row["shoulder_tilt_deg"], row["pelvic_tilt_deg"] = fmt(shoulder), fmt(pelvic)
            self.line(image, ls, rs, red if shoulder > 3 else green)
            self.line(image, lh, rh, red if pelvic > 3 else green)
            left_leg, right_leg = self.angle(p(23), p(25), p(27)), self.angle(p(24), p(26), p(28))
            row["left_leg_angle_deg"], row["right_leg_angle_deg"] = fmt(left_leg), fmt(right_leg)
            for a, b in ((p(23), p(25)), (p(25), p(27)), (p(24), p(26)), (p(26), p(28))): self.line(image, a, b, amber)
            if shoulder > 3: alerts.append("肩線偏移提醒")
            if pelvic > 3: alerts.append("骨盆水平偏移提醒")
        else:
            ear, shoulder, hip, knee, ankle = (7, 11, 23, 25, 27) if view == "左側" else (8, 12, 24, 26, 28)
            ep, sp, hp, kp, ap = p(ear), p(shoulder), p(hip), p(knee), p(ankle)
            neck_proxy, knee_angle = self.angle(ep, sp, sp + np.array([0, -100.])), self.angle(hp, kp, ap)
            row["head_neck_proxy_deg"], row["knee_angle_deg"] = fmt(neck_proxy), fmt(knee_angle)
            self.line(image, ep, sp, amber if neck_proxy > 15 else green)
            self.line(image, hp, kp, amber if knee_angle < 165 else green)
            self.line(image, kp, ap, amber if knee_angle < 165 else green)
            if neck_proxy > 15: alerts.append("頭頸前移指標提醒")
            if knee_angle < 165: alerts.append("膝關節排列提醒")
        row["screening_result"] = "；".join(alerts) if alerts else "本次未見提醒"
        return Analysis(image, row)

    def close(self) -> None: self.pose.close()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Posture Guard｜體態篩檢與追蹤")
        self.geometry("1280x820"); self.minsize(1080, 700)
        self.store, self.analyzer, self.camera = CsvStore(), PostureAnalyzer(), None
        self.camera_loop_running = False
        self.frame: np.ndarray | None = None; self.preview_image = None
        self.session_id, self.step, self.rows = "", 0, []
        self.participant, self.status = tk.StringVar(), tk.StringVar(value="請建立新的匿名檢測。")
        self._style(); self._build(); self.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self) -> None:
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("Title.TLabel", font=("PingFang TC", 24, "bold"), foreground="#153450")
        style.configure("Header.TLabel", font=("PingFang TC", 14, "bold"), foreground="#153450")
        style.configure("TButton", padding=(12, 7)); style.configure("Treeview", rowheight=28)

    def _build(self) -> None:
        header = ttk.Frame(self, padding=(22, 15)); header.pack(fill="x")
        ttk.Label(header, text="Posture Guard", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="標準化姿勢篩檢・趨勢追蹤・專業覆核", foreground="#56748b").pack(side="left", padx=16)
        book = ttk.Notebook(self); book.pack(fill="both", expand=True, padx=18)
        self.home, self.capture_page, self.history, self.review = [ttk.Frame(book, padding=18) for _ in range(4)]
        for page, title in zip((self.home, self.capture_page, self.history, self.review), ("開始檢測", "拍攝與量測", "歷史趨勢", "專業覆核")): book.add(page, text=f"  {title}  ")
        self._home(); self._capture_page(); self._history(); self._review()
        ttk.Label(self, textvariable=self.status, padding=8, relief="groove", anchor="w").pack(fill="x", padx=18, pady=10)

    def _home(self) -> None:
        ttk.Label(self.home, text="建立本次檢測", style="Header.TLabel").pack(anchor="w")
        row = ttk.Frame(self.home, padding=(0, 16)); row.pack(anchor="w")
        ttk.Label(row, text="匿名受測者編號：").pack(side="left")
        ttk.Entry(row, textvariable=self.participant, width=28).pack(side="left", padx=8)
        ttk.Button(row, text="開始新的九張量測", command=self.start_session).pack(side="left")
        text = "拍攝規範\n\n1. 相機固定並保持水平，與受測者距離固定。\n2. 正面、左側、右側各拍三張；全身與雙腳完整入鏡。\n3. 自然站立、目視前方，衣物不要遮住肩與膝。\n4. 只將影像品質合格者納入報告。\n\n本工具為姿勢篩檢與追蹤，不是醫療診斷。"
        ttk.Label(self.home, text=text, justify="left", padding=18, relief="groove").pack(fill="x", pady=10)
        ttk.Label(self.home, text="資料產品化", style="Header.TLabel").pack(anchor="w", pady=(20, 5))
        ttk.Label(self.home, text="每張影像、完整摘要、專業覆核與操作紀錄分別保存於 posture_data 資料夾的 CSV，讓結果可以追溯並用於研究分析。", wraplength=1000).pack(anchor="w")

    def _capture_page(self) -> None:
        top = ttk.Frame(self.capture_page); top.pack(fill="x")
        self.progress = ttk.Label(top, text="尚未建立檢測", style="Header.TLabel"); self.progress.pack(side="left")
        for label, cmd in (("拍攝並分析", self.capture), ("重拍目前位置", self.retake), ("啟動攝影機", self.start_camera)):
            ttk.Button(top, text=label, command=cmd).pack(side="right", padx=4)
        body = ttk.Frame(self.capture_page, padding=(0, 14)); body.pack(fill="both", expand=True)
        self.preview = ttk.Label(body, text="攝影機尚未啟動", anchor="center", relief="sunken"); self.preview.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(body, padding=(16, 0)); side.pack(side="right", fill="y")
        ttk.Label(side, text="本張結果", style="Header.TLabel").pack(anchor="w")
        self.measurement = tk.Text(side, width=40, height=25, wrap="word", state="disabled"); self.measurement.pack(pady=8)

    def _history(self) -> None:
        top = ttk.Frame(self.history); top.pack(fill="x")
        ttk.Label(top, text="歷史檢測紀錄", style="Header.TLabel").pack(side="left")
        ttk.Button(top, text="重新載入 CSV", command=self.load_history).pack(side="right")
        keys = ("date", "participant", "shoulder", "pelvic", "neck", "knee", "quality", "result", "review")
        self.tree = ttk.Treeview(self.history, columns=keys, show="headings", height=13)
        for key, title in zip(keys, ("日期", "匿名編號", "肩線°", "骨盆°", "頭頸指標°", "膝°", "品質", "結果", "覆核")):
            self.tree.heading(key, text=title); self.tree.column(key, width=115, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=12)
        self.chart = tk.Canvas(self.history, height=150, background="white", highlightthickness=1, highlightbackground="#b7c5cf"); self.chart.pack(fill="x")
        self.load_history()

    def _review(self) -> None:
        ttk.Label(self.review, text="專業覆核（研究／臨床合作模式）", style="Header.TLabel").pack(anchor="w")
        ttk.Label(self.review, text="請以預先定義的參考量測方法填寫。覆核資料不會改寫系統原始資料。", wraplength=900).pack(anchor="w", pady=(6, 16))
        form = ttk.Frame(self.review); form.pack(anchor="w")
        self.review_session, self.reviewer, self.reference, self.review_result = tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
        for idx, (label, var) in enumerate((("Session ID", self.review_session), ("覆核者編號", self.reviewer), ("參考方法／量測值", self.reference), ("覆核結論", self.review_result))):
            ttk.Label(form, text=label + "：").grid(row=idx, column=0, sticky="e", pady=5); ttk.Entry(form, textvariable=var, width=55).grid(row=idx, column=1, pady=5)
        ttk.Label(form, text="備註：").grid(row=4, column=0, sticky="ne", pady=5)
        self.review_notes = tk.Text(form, width=54, height=6); self.review_notes.grid(row=4, column=1, pady=5)
        ttk.Button(form, text="儲存覆核紀錄", command=self.save_review).grid(row=5, column=1, sticky="e", pady=10)

    def start_session(self) -> None:
        if not self.participant.get().strip(): messagebox.showwarning("需要匿名編號", "請先輸入匿名受測者編號。"); return
        self.session_id, self.step, self.rows = uuid.uuid4().hex[:12], 0, []
        self.review_session.set(self.session_id); self._progress(); self.store.audit("session_started", self.session_id, self.participant.get().strip())
        self.status.set(f"已建立檢測 {self.session_id}，請到「拍攝與量測」啟動攝影機。")

    def _progress(self) -> None:
        if self.step < len(WORKFLOW):
            view, n = WORKFLOW[self.step]; self.progress.configure(text=f"第 {self.step + 1} / 9 張：{view}，第 {n} 張")
        else: self.progress.configure(text="本次九張量測已完成")

    def start_camera(self) -> None:
        if not self.session_id: messagebox.showinfo("請先建立檢測", "請先建立本次量測。"); return
        if self.camera is None or not self.camera.isOpened(): self.camera = cv2.VideoCapture(0); self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.camera.isOpened(): self.status.set("無法開啟攝影機；請確認相機權限。"); return
        self.status.set("攝影機已啟動。確認全身完整入鏡後拍攝。")
        if not self.camera_loop_running:
            self.camera_loop_running = True
            self.update_camera()

    def update_camera(self) -> None:
        if self.camera is None or not self.camera.isOpened():
            self.camera_loop_running = False
            return
        ok, frame = self.camera.read()
        if ok: self.frame = frame; self.show_image(frame)
        self.after(33, self.update_camera)

    def show_image(self, frame: np.ndarray) -> None:
        image = cv2.flip(frame, 1); h, w = image.shape[:2]; scale = min(720 / w, 560 / h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if ok:
            self.preview_image = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii")); self.preview.configure(image=self.preview_image, text="")

    def capture(self) -> None:
        if self.frame is None or not self.session_id or self.step >= 9: self.status.set("請先建立檢測並啟動攝影機。"); return
        view, count = WORKFLOW[self.step]
        analysis = self.analyzer.analyze(self.frame, view, self.session_id, self.participant.get().strip(), count)
        self.show_image(analysis.image); self.show_measurement(analysis.row)
        if analysis.row["image_quality"] != "合格": self.status.set("影像品質不合格，未納入量測；請調整站位後重拍。"); return
        self.rows.append(analysis.row); self.step += 1; self.store.audit("capture_accepted", self.session_id, f"{view}#{count}")
        if self.step == 9: self.finish_session()
        else: self._progress(); self.status.set("已納入本張量測，請準備下一個視角。")

    def retake(self) -> None:
        if self.session_id and self.step < 9: self.store.audit("retake_requested", self.session_id, str(WORKFLOW[self.step])); self.status.set("目前位置未變更，請重新站位後拍攝。")

    def show_measurement(self, row: dict[str, str]) -> None:
        text = "\n".join((f"品質：{row['image_quality']}", f"說明：{row['quality_note']}", "", f"肩線傾斜：{row['shoulder_tilt_deg'] or '—'}°", f"骨盆傾斜：{row['pelvic_tilt_deg'] or '—'}°", f"頭頸前移代理指標：{row['head_neck_proxy_deg'] or '—'}°", f"膝關節角度：{row['knee_angle_deg'] or '—'}°", "", f"篩檢結果：{row['screening_result']}", "", "注意：頭頸數值為 2D 代理指標，不是臨床 CVA。"))
        self.measurement.configure(state="normal"); self.measurement.delete("1.0", "end"); self.measurement.insert("1.0", text); self.measurement.configure(state="disabled")

    @staticmethod
    def median(rows: list[dict[str, str]], key: str) -> str:
        vals = [float(r[key]) for r in rows if r[key]]; return fmt(float(np.median(vals))) if vals else ""

    def finish_session(self) -> None:
        front, sides = [r for r in self.rows if r["view"] == "正面"], [r for r in self.rows if r["view"] != "正面"]
        alerts = sorted({a for r in self.rows for a in r["screening_result"].split("；") if "提醒" in a})
        summary = {"session_id": self.session_id, "participant_id": self.participant.get().strip(), "measured_at": now(), "app_version": APP_VERSION, "protocol_version": PROTOCOL_VERSION, "front_shoulder_tilt_median_deg": self.median(front, "shoulder_tilt_deg"), "front_pelvic_tilt_median_deg": self.median(front, "pelvic_tilt_deg"), "side_head_neck_proxy_median_deg": self.median(sides, "head_neck_proxy_deg"), "side_knee_median_deg": self.median(sides, "knee_angle_deg"), "quality_pass_rate": "9/9", "screening_result": "；".join(alerts) if alerts else "本次未見提醒", "review_status": "待專業覆核"}
        self.store.append("sessions", self.rows); self.store.append("summaries", [summary]); self.store.audit("session_completed", self.session_id, "9 accepted captures")
        self._progress(); self.status.set("已完成並寫入 CSV，可至「歷史趨勢」查看。"); self.load_history(); messagebox.showinfo("量測完成", "九張合格影像已寫入 CSV。結果僅供篩檢與追蹤，非醫療診斷。")

    def load_history(self) -> None:
        rows = self.store.read("summaries")
        for item in self.tree.get_children(): self.tree.delete(item)
        for r in reversed(rows): self.tree.insert("", "end", values=(r["measured_at"], r["participant_id"], r["front_shoulder_tilt_median_deg"] or "—", r["front_pelvic_tilt_median_deg"] or "—", r["side_head_neck_proxy_median_deg"] or "—", r["side_knee_median_deg"] or "—", r["quality_pass_rate"], r["screening_result"], r["review_status"]))
        self.draw_chart(rows[-12:])

    def draw_chart(self, rows: list[dict[str, str]]) -> None:
        self.chart.delete("all"); width = max(self.chart.winfo_width(), 900); self.chart.create_text(12, 12, anchor="w", text="最近 12 次：肩線傾斜（藍）／骨盆傾斜（橘）", fill="#36556c")
        series = [[float(r[k]) if r.get(k) else None for r in rows] for k in ("front_shoulder_tilt_median_deg", "front_pelvic_tilt_median_deg")]; values = [v for s in series for v in s if v is not None]
        if not values: self.chart.create_text(width / 2, 75, text="尚無可繪製的歷史資料", fill="#789"); return
        maximum = max(5, max(values) * 1.2)
        for s, color in zip(series, ("#2c7fb8", "#e67e22")):
            points = [(35 + i * (width - 60) / max(1, len(rows) - 1), 130 - v / maximum * 90) for i, v in enumerate(s) if v is not None]
            if len(points) > 1: self.chart.create_line(*[q for p in points for q in p], fill=color, width=2, smooth=True)
            for x, y in points: self.chart.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

    def save_review(self) -> None:
        if not self.review_session.get().strip() or not self.reviewer.get().strip(): messagebox.showwarning("資料不足", "請填寫 Session ID 與覆核者編號。"); return
        row = {"session_id": self.review_session.get().strip(), "reviewed_at": now(), "reviewer_id": self.reviewer.get().strip(), "clinical_reference": self.reference.get().strip(), "review_result": self.review_result.get().strip(), "notes": self.review_notes.get("1.0", "end").strip()}
        self.store.append("reviews", [row]); self.store.mark_reviewed(row["session_id"])
        self.store.audit("professional_review_saved", row["session_id"], f"reviewer={row['reviewer_id']}")
        self.load_history(); self.status.set("已儲存覆核紀錄，原始量測不會被改寫。")

    def close(self) -> None:
        if self.camera is not None: self.camera.release()
        self.analyzer.close(); self.destroy()


if __name__ == "__main__": App().mainloop()
