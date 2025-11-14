# app.py
# 🧠 AI Security Guard (ASG) — Video Analysis API

import os, sys, csv, cv2, numpy as np, torch, warnings, uuid, threading, shutil, subprocess
from collections import deque
from flask import Flask, request, jsonify, send_from_directory, abort, url_for
from pytorchvideo.models.hub import slowfast_r50
from ultralytics import YOLO
from os.path import basename
from flask_cors import CORS   # 🔥 추가

# 🔗 DB / Models (User 안 씀)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.models import db, Job, Clip


# ================== Flask & DB 초기화 ==================
app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "server", "instance", "users.db")
DB_PATH = os.path.abspath(DB_PATH)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"



CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}},
    supports_credentials=True,
)

db.init_app(app)

with app.app_context():
    db.create_all()

# ================== (원본) Config 그대로 ==================
CKPT = r".\result\ckpt.pt"
YOLO_WEIGHTS = r".\result\best.pt"
SAVE_VIDEO = True
SAVE_CSV = False

# --- (임시) 비활성화할 클래스들 ---
DISABLE_CLASSES = {"vandalism"}

# --- Detection / ROI gating ---
PERSON_CONF = 0.55
MIN_AREA_RATIO = 0.0035
ASPECT_MIN, ASPECT_MAX = 0.25, 0.90
NO_PERSON_CLEAR = 8
ROI_SCALE = 1.45
ROI_MIN_AREA_RATIO = 0.02
BG_MODE = "gray"  # or "blur"

# --- SlowFast sampling & norm ---
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using torch.load")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

# --- Temporal Stabilization ---
LOGIT_EMA    = 0.70
MARGIN_MIN   = 0.08
SWITCH_DELTA = 0.10
SWITCH_CONSEC = 4
MIN_HOLD = {
    "assault": 32,
    "swoon": 60,
    "trespass": 40,
}
MIN_SHOW_CONF = 0.30

# --- Fallbacks ---
T_FAST_FALLBACK  = 32
ALPHA_FALLBACK   = 4
SIZE_FALLBACK    = 224

# ================== API용 경로 (OUT_DIR를 루트로 사용) ==================
BASE_OUT        = os.path.abspath("./")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 파일 맨 위쪽 설정부
#EVENT_CLIPS_DIR = r"D:\PycharmProjects\pythonProject1\ASG-main\ai\event_clips"
#THUMBS_DIR      = r"D:\PycharmProjects\pythonProject1\ASG-main\ai\thumbnails"

EVENT_CLIPS_DIR = os.path.join(BASE_OUT, "event_clips")
THUMBS_DIR      = os.path.join(BASE_OUT, "thumbnails")
UPLOAD_DIR      = os.path.join(BASE_OUT, "uploads")
ANALYZED_DIR    = os.path.join(BASE_OUT, "analyzed_videos")
CSV_DIR         = os.path.join(BASE_OUT, "csv_logs")

for d in [BASE_OUT, UPLOAD_DIR, ANALYZED_DIR, EVENT_CLIPS_DIR, THUMBS_DIR, CSV_DIR]:
    os.makedirs(d, exist_ok=True)

URL_BASE = "http://127.0.0.1:5001"

# ================== 유틸 함수 ==================
def center_crop_rgb(bgr, size):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256.0 / max(1, min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = max(0, (nh - size) // 2), max(0, (nw - size) // 2)
    img = img[y0:y0+size, x0:x0+size]
    return img


def expand_box(box, scale, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = max(0, int(cx - bw/2)), max(0, int(cy - bh/2))
    nx2, ny2 = min(W, int(cx + bw/2)), min(H, int(cy + bh/2))
    return nx1, ny1, nx2, ny2


def make_roi_frame(frame, dets, scale=1.45, roi_min_ratio=0.02, bg_mode="gray"):
    H, W = frame.shape[:2]
    if not dets:
        return None
    mask = np.zeros((H, W), np.uint8)
    for (x1, y1, x2, y2, *_ ) in dets:
        ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), scale, W, H)
        mask[ey1:ey2, ex1:ex2] = 255
    if mask.mean()/255.0 < roi_min_ratio:
        return None
    if bg_mode == "blur":
        bg = cv2.GaussianBlur(frame, (0, 0), sigmaX=9, sigmaY=9)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m3 = cv2.merge([mask, mask, mask])
    return np.where(m3 == 255, frame, bg)


def yolo_person_boxes(yolo, frame, conf, min_area_ratio, aspect_min, aspect_max):
    r = yolo.predict(source=frame, verbose=False, conf=conf, imgsz=640)[0]
    names = getattr(r, "names", {0: "person"})
    person_ids = {i for i, n in names.items() if str(n).lower() == "person"}
    H, W = frame.shape[:2]
    min_area = min_area_ratio * (H * W)
    out = []
    for b, c, s in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy()
    ):
        if (person_ids and int(c) not in person_ids) and int(c) != 0:
            continue
        x1, y1, x2, y2 = map(float, b)
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        area = w * h
        ar = w / h
        if area < min_area:
            continue
        if not (aspect_min <= ar <= aspect_max):
            continue
        out.append((x1, y1, x2, y2, float(s), int(c)))
    return out

import os
import cv2
import shutil
import subprocess

FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

def extract_clip(video_path, start_s, end_s, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_s = max(0.0, float(start_s))
    end_s   = max(start_s, float(end_s))
    dur     = max(0.10, end_s - start_s)

    # 1) ffmpeg로 먼저 시도 (절대경로 사용)
    if os.path.isfile(FFMPEG_PATH):
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", video_path,                 # 먼저 입력
            "-ss", f"{start_s:.3f}",          # 그 다음 시작 시간
            "-t",  f"{dur:.3f}",
            "-c:v", "libx264",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-loglevel", "error",
            out_path,
        ]
        try:
            rc = subprocess.run(cmd).returncode
            if rc == 0 and os.path.isfile(out_path):
                return True
        except FileNotFoundError:
            # ffmpeg.exe 경로가 잘못됐거나 실행 불가한 경우 → 아래 OpenCV fallback으로
            pass

    # 2) OpenCV fallback
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        cap.release()
        return False

    start_f = int(round(start_s * fps))
    end_f   = int(round(end_s   * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    fidx = start_f
    ok = True
    while fidx < end_f:
        ok2, fr = cap.read()
        if not ok2:
            ok = False
            break
        vw.write(fr)
        fidx += 1

    vw.release()
    cap.release()
    return ok and os.path.isfile(out_path)



def save_thumbnail(video_path, t_sec, out_dir, name_stub):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = int(round(max(0.0, t_sec) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    scale = min(640 / max(1, w), 360 / max(1, h))
    if scale < 1.0:
        fr = cv2.resize(fr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    name = f"{name_stub}_thumb.jpg"
    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, fr)
    return out_path


def fmt_time(sec: float):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_time_cs(sec: float):
    """HH:MM:SS.cc 형식 (clips.start_time 용)"""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


# ================== 전역 모델 핸들 ==================
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SF = None
_CLASSES = None
_T_FAST = None
_ALPHA = None
_SIZE = None
_YOLO = None


def ensure_models_loaded():
    global _SF, _CLASSES, _T_FAST, _ALPHA, _SIZE, _YOLO
    if _YOLO is None:
        _YOLO = YOLO(YOLO_WEIGHTS)
    if _SF is None:
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        _CLASSES = ckpt.get("classes", ["assault", "swoon", "trespass", "vandalism"])
        _T_FAST  = int(ckpt.get("t_fast", T_FAST_FALLBACK))
        _ALPHA   = int(ckpt.get("alpha", ALPHA_FALLBACK))
        _SIZE    = int(ckpt.get("size", SIZE_FALLBACK))
        model = slowfast_r50(pretrained=False)
        model.blocks[-1].proj = torch.nn.Linear(
            model.blocks[-1].proj.in_features,
            len(_CLASSES)
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(_DEVICE)
        _SF = model


# ================== 분석 코어 (DB 연동) ==================
def analyze_core(video_path, job_id: str):
    """
    - stabilized pred_label로 interval 생성
    - 클립/썸네일 생성
    - jobs / clips 테이블에 저장
    """
    ensure_models_loaded()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with app.app_context():
            job_row = Job.query.get(job_id)
            if job_row:
                job_row.status = "failed"
                job_row.progress = 0.0
                job_row.message = f"Cannot open video: {video_path}"
                db.session.commit()
        return

    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("========== DEBUG VIDEO INFO ==========")
    print(f"video_path: {video_path}")
    print(f"fps:          {fps}")
    print(f"total_frames: {total_frames}")
    print(f"W,H:          {W}, {H}")
    print("======================================")

    stem    = os.path.splitext(os.path.basename(video_path))[0]
    out_mp4 = os.path.join(ANALYZED_DIR, f"{stem}_analyze.mp4")
    out_csv = os.path.join(CSV_DIR, f"{stem}_stable.csv")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(out_mp4, fourcc, fps, (W, H)) if SAVE_VIDEO else None

    buf = deque(maxlen=_T_FAST)
    rows = []
    no_person_streak = 0
    frame_idx = 0
    p_ema = None
    current_label = ""
    hold_count = 0
    switch_count = 0

    active = None  # {"label": str, "start_f": int, "start_bbox": [x1,y1,x2,y2] | None}
    intervals = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # 진행률 갱신 (10프레임마다)
            if total_frames > 0 and frame_idx % 10 == 0:
                print(f"[DEBUG] progress trigger: frame_idx={frame_idx}, total_frames={total_frames}")
                with app.app_context():
                    job_row = Job.query.get(job_id)
                    if job_row:
                        job_row.progress = round(100.0 * frame_idx / total_frames, 2)
                        job_row.status = "running"
                        job_row.message = "processing"
                        db.session.commit()

            dets = yolo_person_boxes(_YOLO, frame, PERSON_CONF, MIN_AREA_RATIO, ASPECT_MIN, ASPECT_MAX)
            roi  = make_roi_frame(frame, dets, scale=ROI_SCALE, roi_min_ratio=ROI_MIN_AREA_RATIO, bg_mode=BG_MODE)

            # === ROI 없을 때: normal로 표시 ===
            if roi is None:
                no_person_streak += 1
                if no_person_streak >= NO_PERSON_CLEAR:
                    buf.clear()
                    p_ema = None
                    current_label = ""
                    hold_count = 0
                    switch_count = 0

                vis = frame.copy()
                cv2.putText(vis, "normal", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                for (x1, y1, x2, y2, *_ ) in dets:
                    cv2.rectangle(vis, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                if SAVE_VIDEO and writer is not None:
                    writer.write(vis)
                if SAVE_CSV:
                    rows.append({"frame": frame_idx, "pred": "normal", "conf": ""})
                continue
            else:
                no_person_streak = 0

            rgb = center_crop_rgb(roi, _SIZE)
            buf.append(rgb)
            pred_label = ""
            pred_conf  = ""

            if len(buf) == _T_FAST:
                fast = torch.from_numpy(
                    np.stack(list(buf), 0)
                ).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
                slow = fast[:, :, ::_ALPHA, :, :]
                fast = (fast - MEAN) / STD
                slow = (slow - MEAN) / STD

                with torch.no_grad():
                    logits = _SF([slow.to(_DEVICE), fast.to(_DEVICE)])
                    p = torch.softmax(logits, dim=1)[0].cpu().numpy()

                mask = np.ones_like(p, dtype=np.float32)
                for i, cls in enumerate(_CLASSES):
                    if cls in DISABLE_CLASSES:
                        mask[i] = 0.0
                p = p * mask
                s = float(p.sum())
                if s > 1e-8:
                    p /= s
                else:
                    p[:] = 0.0

                if p_ema is None:
                    p_ema = p.copy()
                else:
                    p_ema = LOGIT_EMA * p_ema + (1.0 - LOGIT_EMA) * p

                order = np.argsort(-p_ema)
                k1 = int(order[0])
                k2 = int(order[1] if len(order) > 1 else order[0])
                top1, top2 = float(p_ema[k1]), float(p_ema[k2])
                label1 = _CLASSES[k1]

                if current_label == "":
                    if top1 >= MIN_SHOW_CONF and (top1 - top2) >= MARGIN_MIN:
                        current_label = label1
                        hold_count = 1
                        switch_count = 0
                else:
                    if label1 == current_label:
                        hold_count += 1
                        switch_count = 0
                    else:
                        curr_idx = _CLASSES.index(current_label)
                        cond_delta  = (top1 - float(p_ema[curr_idx])) >= SWITCH_DELTA
                        cond_margin = (top1 - top2) >= MARGIN_MIN
                        cond_hold   = hold_count >= MIN_HOLD.get(current_label, 16)
                        if cond_delta and cond_margin and cond_hold:
                            switch_count += 1
                            if switch_count >= SWITCH_CONSEC:
                                current_label = label1
                                hold_count = 1
                                switch_count = 0
                        else:
                            switch_count = 0

                if current_label and top1 >= MIN_SHOW_CONF:
                    pred_label = current_label
                    pred_conf  = f"{top1:.2f}"

            # interval on/off
            first_bbox = None
            if dets:
                x1, y1, x2, y2, *_ = dets[0]
                first_bbox = [int(x1), int(y1), int(x2), int(y2)]

            if pred_label:
                if active is None:
                    active = {
                        "label": pred_label,
                        "start_f": max(0, frame_idx - _T_FAST + 1),
                        "start_bbox": first_bbox,
                    }
                else:
                    if active["label"] != pred_label:
                        s = active["start_f"] / fps
                        e = frame_idx / fps
                        intervals.append({
                            "start": s,
                            "end": e,
                            "label": active["label"],
                            "bbox": active["start_bbox"],
                        })
                        active = {
                            "label": pred_label,
                            "start_f": max(0, frame_idx - _T_FAST + 1),
                            "start_bbox": first_bbox,
                        }
            else:
                if active is not None:
                    s = active["start_f"] / fps
                    e = frame_idx / fps
                    intervals.append({
                        "start": s,
                        "end": e,
                        "label": active["label"],
                        "bbox": active["start_bbox"],
                    })
                    active = None

            # 오버레이/저장
            vis = frame.copy()
            # pred_label 없으면 화면에는 normal로
            title = f"{pred_label} {pred_conf}" if pred_label else "normal"

            for (x1, y1, x2, y2, *_ ) in dets:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, title, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            if SAVE_VIDEO and writer is not None:
                writer.write(vis)
            if SAVE_CSV:
                rows.append({
                    "frame": frame_idx,
                    "pred": pred_label if pred_label else "normal",
                    "conf": pred_conf
                })

        # 루프 종료 후 active 마무리
        if active is not None:
            s = active["start_f"] / fps
            e = frame_idx / fps
            intervals.append({
                "start": s,
                "end": e,
                "label": active["label"],
                "bbox": active["start_bbox"],
            })

    except Exception as e:
        with app.app_context():
            job_row = Job.query.get(job_id)
            if job_row:
                job_row.status = "failed"
                job_row.message = str(e)
                db.session.commit()
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # CSV 저장
    if SAVE_CSV and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "pred", "conf"])
            w.writeheader()
            w.writerows(rows)

    # 클립 / 썸네일 생성
    clips_meta = []
    for i, it in enumerate(intervals, 1):
        base = os.path.splitext(os.path.basename(video_path))[0]
        clip_name = f"{base}_clip{i}.mp4"
        clip_path = os.path.join(EVENT_CLIPS_DIR, clip_name)
        ok = extract_clip(video_path, it["start"], it["end"], clip_path)

        mid_t = (it["start"] + it["end"]) / 2.0
        thumb_stub = f"{base}_clip{i}"
        thumb_path = save_thumbnail(video_path, mid_t, THUMBS_DIR, thumb_stub)

        bbox = it.get("bbox")
        clips_meta.append({
            "ok": ok,
            "label": it["label"],
            "start_sec": it["start"],
            "bbox": bbox,
            "clip_name": clip_name,
            "clip_path": clip_path,
            "thumbnail": thumb_path,
        })

    # DB 반영
    with app.app_context():
        job_row = Job.query.get(job_id)
        if not job_row:
            return

        if SAVE_VIDEO and os.path.isfile(out_mp4):
            job_row.annotated_video = os.path.join(
                "/analyzed_videos",
                os.path.basename(out_mp4)
            )
        job_row.status   = "done"
        job_row.progress = 100.0
        job_row.message  = "completed"

        for meta in clips_meta:
            if not meta["ok"]:
                continue
            bbox = meta["bbox"]
            start_x = start_y = start_w = start_h = None
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                start_x, start_y = int(x1), int(y1)
                start_w, start_h = int(x2 - x1), int(y2 - y1)

            clip_row = Clip(
                job_id     = job_id,
                class_name = meta["label"],
                checked    = False,
                start_time = fmt_time_cs(meta["start_sec"]),
                start_x    = start_x,
                start_y    = start_y,
                start_w    = start_w,
                start_h    = start_h,
                clip_name  = meta["clip_name"],
                clip_path  = meta["clip_path"],
                thumbnail  = meta["thumbnail"],
            )
            db.session.add(clip_row)

        db.session.commit()


# ================== Flask Routes ==================
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    업로드 또는 JSON {"video_path": "...", "username": "..."} 둘 다 지원
    응답: {job_id, status, progress, video_path, username}
    """
    try:
        video_path = None
        username   = None

        # form-data 업로드
        if "video" in request.files:
            f = request.files["video"]
            if not f.filename:
                return jsonify({"error": "Empty file name"}), 400
            save_to = os.path.join(UPLOAD_DIR, f.filename)
            f.save(save_to)
            video_path = save_to
            username   = request.form.get("username")

        # JSON 요청
        if video_path is None:
            data = request.get_json(silent=True) or {}
            video_path = data.get("video_path")
            username   = username or data.get("username")

        if not video_path or not os.path.isfile(video_path):
            return jsonify({
                "error": "Provide a valid video via form-data 'video' or JSON {'video_path': '...'}"
            }), 400

        if not username:
            username = "guest"

        # DB에 Job 생성 (username 문자열만 저장)
        with app.app_context():
            job_id = str(uuid.uuid4())
            job_row = Job(
                job_id          = job_id,
                username        = username,      # models.Job에 username 컬럼 있어야 함
                video_path      = video_path,
                status          = "running",
                progress        = 0.0,
                annotated_video = None,
                message         = "started",
            )
            db.session.add(job_row)
            db.session.commit()

        # 백그라운드 분석 시작
        th = threading.Thread(target=analyze_core, args=(video_path, job_id), daemon=True)
        th.start()

        return jsonify({
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "video_path": video_path,
            "username": username,
        })

    except Exception as e:
        import traceback
        print("\n[ERROR] /analyze 내부에서 예외 발생:")
        traceback.print_exc()  # 터미널에 전체 스택 출력
        return jsonify({
            "error": "internal_error",
            "detail": str(e),
        }), 500


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    with app.app_context():
        job_row = Job.query.get(job_id)
        if not job_row:
            return jsonify({"error": "job_id not found"}), 404
        # Job.to_dict(include_clips=False) 안에 username 포함돼 있어야 함
        return jsonify(job_row.to_dict(include_clips=False))


import os

@app.route("/jobs/<job_id>/clips", methods=["GET"])
def get_clips_by_job(job_id):
    job = Job.query.filter_by(job_id=job_id).first()
    if not job:
        return jsonify({"detail": "Job not found"}), 404

    clips = Clip.query.filter_by(job_id=job_id).order_by(Clip.start_time).all()
    result = {
        "job_id": job.job_id,
        "video_path": job.video_path,
        "count": len(clips),
        "clips": [],
    }

    from os.path import basename

    for c in clips:
        d = c.to_dict()

        d["checked"] = bool(getattr(c, "checked", 0))

        # 🔹 1) xywh → (x1,y1,x2,y2) start_bbox 로 변환
        x = c.start_x   # ← 여기를 네 실제 컬럼명으로
        y = c.start_y   # ← 예: c.x, c.y, c.start_x, c.start_y 등
        w = c.start_w   # ← 예: c.w, c.width
        h = c.start_h   # ← 예: c.h, c.height

        if None not in (x, y, w, h):
            d["start_bbox"] = {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
            }
        else:
            d["start_bbox"] = None

        # 🔹 2) 클립 URL
        clip_name = d.get("clip_name")
        if clip_name:
            d["clip_url"] = url_for("serve_clip", fname=clip_name, _external=False)
        else:
            d["clip_url"] = None

        # 🔹 3) 썸네일 URL
        thumb_name = d.get("thumbnail") or d.get("thumb_path")
        if thumb_name:
            d["thumb_url"] = url_for(
                "serve_thumb", fname=basename(thumb_name), _external=False
            )
        else:
            d["thumb_url"] = None

        result["clips"].append(d)

    return jsonify(result), 200


# 파일 서빙
@app.route("/event_clips/<path:fname>", methods=["GET"])
def serve_clip(fname):
    path = os.path.join(EVENT_CLIPS_DIR, fname)
    print("[serve_clip] path =", path, "exists:", os.path.isfile(path))

    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(EVENT_CLIPS_DIR, fname, as_attachment=False)



@app.route("/thumbnails/<path:fname>", methods=["GET"])
def serve_thumb(fname):
    path = os.path.join(THUMBS_DIR, fname)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(THUMBS_DIR, fname, as_attachment=False, mimetype="image/jpeg")


@app.route("/analyzed_videos/<path:fname>", methods=["GET"])
def serve_analyzed(fname):
    path = os.path.join(ANALYZED_DIR, fname)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(ANALYZED_DIR, fname, as_attachment=False)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"name": "AI Security Guard (ASG)", "status": "ok", "base_url": URL_BASE})

@app.route("/clips/<int:clip_id>/check", methods=["PATCH"])
def mark_clip_checked(clip_id):
    clip = Clip.query.get(clip_id)
    if not clip:
        return jsonify({"error": "Clip not found"}), 404
    clip.checked = True

    db.session.commit()
    return jsonify(
        {"message": "checked set to true", "clip_id": clip_id, "checked": True}
    )

@app.route("/jobs/latest", methods=["GET"])
def get_latest_job_for_user():
    username = request.args.get("username")
    if not username:
        return jsonify({"detail": "username is required"}), 400

    # jobs.db 에서 이 사용자걸 최신순으로 하나만
    conn = db.connect("jobs.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT job_id FROM jobs WHERE username = ? ORDER BY rowid DESC LIMIT 1",
        (username,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"detail": "no jobs for this user"}), 404

    return jsonify({"job_id": row[0]}), 200


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    app.run(host="127.0.0.1", port=5001, debug=True)
