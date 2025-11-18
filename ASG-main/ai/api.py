import os, sys, csv, cv2, numpy as np, torch, warnings, uuid, threading, shutil, subprocess
from collections import deque
from flask import Flask, request, jsonify, send_from_directory, abort, url_for
from pytorchvideo.models.hub import slowfast_r50
from ultralytics import YOLO
from os.path import basename
from flask_cors import CORS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.models import db, Job, Clip


# Flask & DB 초기화
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

# Config
CKPT = r".\result\ckpt.pt"
YOLO_WEIGHTS = r".\result\best.pt"
SAVE_VIDEO = True

# (임시)비활성화할 클래스들
DISABLE_CLASSES = {"vandalism"}

# ffmpeg 절대경로 (필요시 수정하세요)
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# Detection / ROI gating
PERSON_CONF = 0.55
MIN_AREA_RATIO = 0.0035
ASPECT_MIN, ASPECT_MAX = 0.25, 0.90
NO_PERSON_CLEAR = 8
ROI_SCALE = 1.45
ROI_MIN_AREA_RATIO = 0.02
BG_MODE = "gray"  # or "blur"

# SlowFast sampling & norm
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using torch.load")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)

# Temporal Stabilization
LOGIT_EMA    = 0.70
MARGIN_MIN   = 0.10
SWITCH_DELTA = 0.12
SWITCH_CONSEC = 5
MIN_HOLD = {
    "assault": 48,
    "swoon": 90,
    "trespass": 40,
}
MIN_SHOW_CONF = 0.30

# Fallbacks
T_FAST_FALLBACK  = 32
ALPHA_FALLBACK   = 4
SIZE_FALLBACK    = 224

# Rule pack: assault / trespass
CLASS_SCALE = {"assault": 1.10, "trespass": 1.00, "swoon": 1.00}

# Assault
ASSAULT_MIN_PEOPLE   = 2
ASSAULT_NEAR_THRESH  = 0.15
ASSAULT_MOTION_GATE  = 7.0
ASSAULT_BOOST        = 1.50
ASSAULT_DAMP_SINGLE  = 0.60
KICK_RATIO_THRESH    = 1.35
KICK_MOTION_THRESH   = 8.0
KICK_BOOST           = 1.25
SWING_EDGE_THRESH    = 12.0
SWING_NEAR_THRESH    = 0.18
SWING_BOOST          = 1.20
QUIET_MOTION_THRESH  = 5.0
WEAK_KICK_RATIO      = 1.10
WEAK_SWING           = 8.0
SOLO_SUPPRESS_ASSAULT= 0.25
ASSAULT_CONTRA_FRAMES = 10
ASSAULT_SUPPRESS_TRESPASS = 0.80

# Trespass (no-zone): relax thresholds
EDGE_MARGIN_RATIO       = 0.08
MIN_OUTSIDE_FOR_ENTRY_FR= 8
CENTRAL_MARGIN_RATIO    = 0.22
TRESPASS_STAY_FR        = 12
TRESPASS_BOOST          = 1.35
TRESPASS_DAMP_WANDER    = 0.80
MIN_INWARD_SPEED_NORM   = 0.0025
MIN_INWARD_DEPTH_RATIO  = 0.12
ENTRY_TIMEOUT_FR        = 75

# Pre-entry loiter → entry boost
LOITER_WINDOW_FR        = 45
LOITER_EDGE_BAND        = 0.12
LOITER_RADIUS_NORM      = 0.04
LOITER_ENTRY_BOOST      = 1.25

# Fence jump heuristic
JUMP_WIN_FR             = 16
JUMP_VY_SPIKE           = 0.018
JUMP_TOTAL_DY           = 0.08
JUMP_ENTRY_BOOST        = 1.30
JUMP_STAY_RELAX         = 10

# NEW: "새로 등장" 진입(문 열고 들어오기 등)
NEW_APPEAR_WINDOW_FR    = 30
NEW_APPEAR_CEN_REQ      = 4
NEW_APPEAR_ENTRY_BOOST  = 1.20


# API용 경로
BASE_OUT        = os.path.abspath("./")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENT_CLIPS_DIR = os.path.join(BASE_OUT, "event_clips")
THUMBS_DIR      = os.path.join(BASE_OUT, "thumbnails")
UPLOAD_DIR      = os.path.join(BASE_OUT, "uploads")
ANALYZED_DIR    = os.path.join(BASE_OUT, "analyzed_videos")

for d in [BASE_OUT, UPLOAD_DIR, ANALYZED_DIR, EVENT_CLIPS_DIR, THUMBS_DIR]:
    os.makedirs(d, exist_ok=True)

URL_BASE = "http://127.0.0.1:5001"

# 유틸 함수
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

def extract_clip(video_path, start_s, end_s, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    start_s = max(0.0, float(start_s))
    end_s   = max(start_s, float(end_s))
    dur     = max(0.10, end_s - start_s)
    if os.path.isfile(FFMPEG_PATH):
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-ss", f"{start_s:.3f}",
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
            pass
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

def reencode_to_h264(in_path: str):
    if not (os.path.isfile(FFMPEG_PATH) and os.path.isfile(in_path)):
        return
    tmp_out = in_path + ".h264.mp4"
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-loglevel", "error",
        tmp_out,
    ]
    rc = subprocess.run(cmd).returncode
    if rc == 0 and os.path.isfile(tmp_out):
        os.replace(tmp_out, in_path)

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
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"

# 전역 모델 핸들
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

# 분석
def analyze_core(video_path, job_id: str):
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
    stem    = os.path.splitext(os.path.basename(video_path))[0]
    out_mp4 = os.path.join(ANALYZED_DIR, f"{stem}_analyze.mp4")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(out_mp4, fourcc, fps, (W, H)) if SAVE_VIDEO else None
    buf = deque(maxlen=_T_FAST)
    no_person_streak = 0
    frame_idx = 0
    p_ema = None
    current_label = ""
    hold_count = 0
    switch_count = 0
    last_gray_rule = None
    last_edge_rule = None
    last_mean_center = None
    outside_streak = 0
    entry_armed = False
    entry_happened = False
    armed_edge = None
    armed_center = None
    armed_frames = 0
    central_streak = 0
    edge_centers = deque(maxlen=LOITER_WINDOW_FR)
    vy_buffer = deque(maxlen=JUMP_WIN_FR)
    last_person_count = 0
    spawn_armed = False
    spawn_timer = 0
    assault_contra = 0
    solo_suppressed = False
    active = None
    intervals = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if total_frames > 0 and frame_idx % 10 == 0:
                with app.app_context():
                    job_row = Job.query.get(job_id)
                    if job_row:
                        job_row.progress = round(100.0 * frame_idx / total_frames, 2)
                        job_row.status = "running"
                        job_row.message = "processing"
                        db.session.commit()

            dets = yolo_person_boxes(_YOLO, frame, PERSON_CONF, MIN_AREA_RATIO, ASPECT_MIN, ASPECT_MAX)
            roi  = make_roi_frame(frame, dets, scale=ROI_SCALE, roi_min_ratio=ROI_MIN_AREA_RATIO, bg_mode=BG_MODE)

            if roi is None:
                no_person_streak += 1
                if no_person_streak >= NO_PERSON_CLEAR:
                    buf.clear()
                    p_ema = None
                    current_label = ""
                    hold_count = 0
                    switch_count = 0
                    last_gray_rule = None
                    last_edge_rule = None
                    last_mean_center = None
                    outside_streak = 0
                    entry_armed = False
                    entry_happened = False
                    armed_edge = None
                    armed_center = None
                    armed_frames = 0
                    central_streak = 0
                    edge_centers.clear()
                    vy_buffer.clear()
                    last_person_count = 0
                    spawn_armed = False
                    spawn_timer = 0
                    assault_contra = 0
                    solo_suppressed = False

                vis = frame.copy()
                cv2.putText(vis, "normal", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                for (x1, y1, x2, y2, *_ ) in dets:
                    cv2.rectangle(vis, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                if SAVE_VIDEO and writer is not None:
                    writer.write(vis)
                continue
            else:
                no_person_streak = 0

            Hf, Wf = frame.shape[:2]
            diag = (Hf**2 + Wf**2) ** 0.5

            centers = []
            for (x1, y1, x2, y2, *_ ) in dets:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centers.append((cx, cy))

            mean_center = None
            if centers:
                mean_center = (
                    float(np.mean([c[0] for c in centers])),
                    float(np.mean([c[1] for c in centers])),
                )

            v = (0.0, 0.0)
            vy_norm = 0.0
            if mean_center is not None and last_mean_center is not None:
                dx = mean_center[0] - last_mean_center[0]
                dy = mean_center[1] - last_mean_center[1]
                v = (dx, dy)
                vy_norm = dy / max(1.0, Hf)
            last_mean_center = mean_center

            min_pair_dist = 1.0
            if len(centers) >= 2:
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        d = (dx * dx + dy * dy) ** 0.5 / max(1.0, diag)
                        if d < min_pair_dist:
                            min_pair_dist = d

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            motion_val = 0.0
            diff = None
            if last_gray_rule is not None and last_gray_rule.shape == roi_gray.shape:
                diff = cv2.absdiff(roi_gray, last_gray_rule)
                motion_val = float(np.mean(diff))
            last_gray_rule = roi_gray

            kick_ratio = 0.0
            lower_val = 0.0
            upper_val = 0.0
            if diff is not None:
                h2 = diff.shape[0] // 2
                upper_val = float(np.mean(diff[:h2, :]))
                lower_val = float(np.mean(diff[h2:, :]))
                kick_ratio = lower_val / max(1e-5, upper_val)

            swing_val = 0.0
            edges = cv2.Canny(roi_gray, 50, 150)
            if last_edge_rule is not None and last_edge_rule.shape == edges.shape:
                ediff = cv2.absdiff(edges, last_edge_rule)
                h2 = edges.shape[0] // 2
                swing_val = float(np.mean(ediff[:h2, :]))
            last_edge_rule = edges

            edge_margin_w = EDGE_MARGIN_RATIO * Wf
            edge_margin_h = EDGE_MARGIN_RATIO * Hf
            near_edge = False
            nearest_edge = None
            if mean_center is not None:
                x, y = mean_center
                dists = {"left": x, "right": Wf - x, "top": y, "bottom": Hf - y}
                nearest_edge = min(dists, key=dists.get)
                if (
                    (x < edge_margin_w)
                    or (x > Wf - edge_margin_w)
                    or (y < edge_margin_h)
                    or (y > Hf - edge_margin_h)
                ):
                    near_edge = True

            if mean_center is not None:
                x, y = mean_center
                if (
                    (x < LOITER_EDGE_BAND * Wf)
                    or (x > (1 - LOITER_EDGE_BAND) * Wf)
                    or (y < LOITER_EDGE_BAND * Hf)
                    or (y > (1 - LOITER_EDGE_BAND) * Hf)
                ):
                    edge_centers.append((x, y))
                else:
                    edge_centers.clear()

            if near_edge:
                outside_streak += 1
                if outside_streak >= MIN_OUTSIDE_FOR_ENTRY_FR:
                    entry_armed = True
                    armed_edge = nearest_edge
                    armed_center = mean_center
                    armed_frames = 0
            else:
                outside_streak = 0

            if entry_armed:
                armed_frames += 1
                if armed_frames > ENTRY_TIMEOUT_FR:
                    entry_armed = False
                    armed_edge = None
                    armed_center = None
                    armed_frames = 0
                    entry_happened = False

            if (
                entry_armed
                and mean_center is not None
                and armed_center is not None
                and armed_edge is not None
            ):
                normals = {"left": (1, 0), "right": (-1, 0), "top": (0, 1), "bottom": (0, -1)}
                nx, ny = normals[armed_edge]
                inward_speed = ((v[0] * nx + v[1] * ny) / max(1.0, diag))
                depth_pix = (mean_center[0] - armed_center[0]) * nx + (
                    mean_center[1] - armed_center[1]
                ) * ny
                denom = (Wf if armed_edge in ("left", "right") else Hf)
                depth_ratio = (depth_pix / denom) if depth_pix > 0 else 0.0
                if (
                    inward_speed >= MIN_INWARD_SPEED_NORM
                    and depth_ratio >= MIN_INWARD_DEPTH_RATIO
                ):
                    entry_happened = True
                    entry_armed = False
                    armed_edge = None
                    armed_center = None
                    armed_frames = 0

            central = False
            if mean_center is not None:
                cx_, cy_ = mean_center
                cmx = CENTRAL_MARGIN_RATIO * Wf
                cmy = CENTRAL_MARGIN_RATIO * Hf
                if (cmx <= cx_ <= Wf - cmx) and (cmy <= cy_ <= Hf - cmy):
                    central = True
            if central and entry_happened:
                central_streak += 1
            else:
                central_streak = max(0, central_streak - 1)

            loiter_boost = 1.0
            if len(edge_centers) >= LOITER_WINDOW_FR:
                xs = np.array([p[0] for p in edge_centers])
                ys = np.array([p[1] for p in edge_centers])
                cx_, cy_ = xs.mean(), ys.mean()
                rad = np.mean(
                    np.sqrt((xs - cx_) ** 2 + (ys - cy_) ** 2)
                ) / max(1.0, np.sqrt(Wf * Wf + Hf * Hf))
                if rad >= LOITER_RADIUS_NORM:
                    loiter_boost = LOITER_ENTRY_BOOST

            jump_boost = 1.0
            if mean_center is not None:
                vy_buffer.append(vy_norm)
                if len(vy_buffer) >= 4:
                    total_dy = abs(sum(vy_buffer))
                    max_vy = max(abs(x) for x in vy_buffer)
                    if max_vy >= JUMP_VY_SPIKE and total_dy >= JUMP_TOTAL_DY:
                        jump_boost = JUMP_ENTRY_BOOST

            person_count = len(dets)
            if person_count > last_person_count:
                spawn_armed = True
                spawn_timer = 0
            last_person_count = person_count
            if spawn_armed:
                spawn_timer += 1
                if spawn_timer > NEW_APPEAR_WINDOW_FR:
                    spawn_armed = False

            spawn_boost = 1.0
            if spawn_armed and central_streak >= NEW_APPEAR_CEN_REQ:
                entry_happened = True
                spawn_boost = NEW_APPEAR_ENTRY_BOOST
                spawn_armed = False

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

                for i, cls in enumerate(_CLASSES):
                    p[i] *= CLASS_SCALE.get(cls, 1.0)

                # Assault 규칙
                assault_like = False
                if "assault" in _CLASSES:
                    ia = _CLASSES.index("assault")
                    base_cond = (
                        len(centers) >= ASSAULT_MIN_PEOPLE
                        and min_pair_dist <= ASSAULT_NEAR_THRESH
                        and motion_val >= ASSAULT_MOTION_GATE
                    )
                    kick_cond = (
                        kick_ratio >= KICK_RATIO_THRESH
                        and lower_val >= KICK_MOTION_THRESH
                    )
                    swing_cond = (
                        swing_val >= SWING_EDGE_THRESH
                        and min_pair_dist <= SWING_NEAR_THRESH
                    )
                    if base_cond:
                        p[ia] *= ASSAULT_BOOST
                        assault_like = True
                    if kick_cond:
                        p[ia] *= KICK_BOOST
                        assault_like = True
                    if swing_cond:
                        p[ia] *= SWING_BOOST
                        assault_like = True
                    if (
                        len(centers) < 2
                        and motion_val < QUIET_MOTION_THRESH
                        and kick_ratio < WEAK_KICK_RATIO
                        and swing_val < WEAK_SWING
                    ):
                        p[ia] *= SOLO_SUPPRESS_ASSAULT
                        solo_suppressed = True

                # Trespass 규칙
                if "trespass" in _CLASSES:
                    it = _CLASSES.index("trespass")
                    if entry_happened and central_streak >= TRESPASS_STAY_FR:
                        p[it] *= TRESPASS_BOOST * loiter_boost * jump_boost * spawn_boost
                    else:
                        p[it] *= TRESPASS_DAMP_WANDER
                    if assault_like:
                        p[it] *= ASSAULT_SUPPRESS_TRESPASS

                # 정규화
                s = float(p.sum())
                if s > 1e-8:
                    p /= s
                else:
                    p[:] = 0.0

                # EMA
                if p_ema is None:
                    p_ema = p.copy()
                else:
                    p_ema = LOGIT_EMA * p_ema + (1.0 - LOGIT_EMA) * p

                # assault 모순 해제
                if "assault" in _CLASSES:
                    if (
                        (len(centers) < 2 or min_pair_dist > 0.30)
                        and motion_val < QUIET_MOTION_THRESH
                    ):
                        assault_contra += 1
                    else:
                        assault_contra = 0
                    if current_label == "assault" and assault_contra >= ASSAULT_CONTRA_FRAMES:
                        current_label = ""
                        hold_count = 0
                        switch_count = 0
                        assault_contra = 0

                # 히스테리시스
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
                        s_int = active["start_f"] / fps
                        e_int = frame_idx / fps
                        intervals.append({
                            "start": s_int,
                            "end": e_int,
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
                    s_int = active["start_f"] / fps
                    e_int = frame_idx / fps
                    intervals.append({
                        "start": s_int,
                        "end": e_int,
                        "label": active["label"],
                        "bbox": active["start_bbox"],
                    })
                    active = None

            vis = frame.copy()
            title = f"{pred_label} {pred_conf}" if pred_label else "normal"

            for (x1, y1, x2, y2, *_ ) in dets:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, title, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            if SAVE_VIDEO and writer is not None:
                writer.write(vis)

        if active is not None:
            s_int = active["start_f"] / fps
            e_int = frame_idx / fps
            intervals.append({
                "start": s_int,
                "end": e_int,
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

    if SAVE_VIDEO and os.path.isfile(out_mp4):
        reencode_to_h264(out_mp4)


    clips_meta = []
    for i, it in enumerate(intervals, 1):
        base = os.path.splitext(os.path.basename(video_path))[0]
        clip_name = f"{base}_clip{i}.mp4"
        clip_path = os.path.join(EVENT_CLIPS_DIR, clip_name)

        ok = extract_clip(out_mp4, it["start"], it["end"], clip_path)

        mid_t = (it["start"] + it["end"]) / 2.0
        thumb_stub = f"{base}_clip{i}"
        thumb_path = save_thumbnail(out_mp4, mid_t, THUMBS_DIR, thumb_stub)

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


# Flask Routes
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        video_path = None
        username   = None
        if "video" in request.files:
            f = request.files["video"]
            if not f.filename:
                return jsonify({"error": "Empty file name"}), 400
            save_to = os.path.join(UPLOAD_DIR, f.filename)
            f.save(save_to)
            video_path = save_to
            username   = request.form.get("username")
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
        with app.app_context():
            job_id = str(uuid.uuid4())
            job_row = Job(
                job_id          = job_id,
                username        = username,
                video_path      = video_path,
                status          = "running",
                progress        = 0.0,
                annotated_video = None,
                message         = "started",
            )
            db.session.add(job_row)
            db.session.commit()
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
        traceback.print_exc()
        return jsonify({
            "error": "internal_error",
            "detail": str(e),
        }), 500

@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job_row = Job.query.get(job_id)
    if not job_row:
        return jsonify({"error": "job_id not found"}), 404
    data = job_row.to_dict(include_clips=False)
    annotated_path = (
        data.get("annotated_path")
        or data.get("annotated_video")
        or data.get("analyzed_video_path")
        or getattr(job_row, "annotated_path", None)
        or getattr(job_row, "annotated_video", None)
        or getattr(job_row, "analyzed_video_path", None)
    )
    if annotated_path:
        fname = basename(str(annotated_path))
        data["annotated_video_url"] = url_for(
            "serve_analyzed",
            fname=fname,
            _external=False,
        )
    else:
        data["annotated_video_url"] = None
    return jsonify(data), 200

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
    for c in clips:
        d = c.to_dict()
        d["checked"] = bool(getattr(c, "checked", 0))
        x = c.start_x
        y = c.start_y
        w = c.start_w
        h = c.start_h
        if None not in (x, y, w, h):
            d["start_bbox"] = {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
            }
        else:
            d["start_bbox"] = None
        clip_name = d.get("clip_name")
        if clip_name:
            d["clip_url"] = url_for("serve_clip", fname=clip_name, _external=False)
        else:
            d["clip_url"] = None
        thumb_name = d.get("thumbnail") or d.get("thumb_path")
        if thumb_name:
            d["thumb_url"] = url_for(
                "serve_thumb", fname=basename(thumb_name), _external=False
            )
        else:
            d["thumb_url"] = None
        result["clips"].append(d)
    return jsonify(result), 200

@app.route("/event_clips/<path:fname>", methods=["GET"])
def serve_clip(fname):
    path = os.path.join(EVENT_CLIPS_DIR, fname)
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
    try:
        q = Job.query
        if hasattr(Job, "created_at"):
            q = q.order_by(Job.created_at.desc())
        elif hasattr(Job, "id"):
            q = q.order_by(Job.id.desc())
        else:
            q = q.order_by(Job.job_id.desc())
        job_row = q.first()
        if not job_row:
            return jsonify({"error": "no jobs"}), 404
        return jsonify({"job_id": job_row.job_id}), 200
    except Exception as e:
        print("[/jobs/latest] ERROR:", e)
        return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    app.run(host="127.0.0.1", port=5001, debug=True)
