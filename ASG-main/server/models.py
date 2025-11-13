# server/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import relationship

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    password = Column(String(200), nullable=False)
    video = Column(String(200), nullable=True)
    original_video = Column(String, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "video": self.video,
        }


class Job(db.Model):
    __tablename__ = "jobs"

    # ASG에서 쓰는 UUID 문자열
    job_id = Column(String, primary_key=True)

    # ✅ User 테이블이랑 FK 안 걸고, 그냥 이름 문자열만 저장
    username = Column(String(80), nullable=False, index=True)

    video_path = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending/running/done/failed
    progress = Column(Float, nullable=False, default=0.0)
    annotated_video = Column(String(255), nullable=True)
    message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Job 1 : N Clip
    clips = relationship(
        "Clip",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self, include_clips: bool = False):
        data = {
            "job_id": self.job_id,
            "username": self.username,
            "video_path": self.video_path,
            "status": self.status,
            "progress": self.progress,
            "annotated_video": self.annotated_video,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_clips:
            data["clips"] = [c.to_dict() for c in self.clips]
        return data


class Clip(db.Model):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # jobs.job_id 에 FK
    job_id = Column(String, ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False, index=True)

    class_name = Column(String(50), nullable=False)   # assault / swoon / trespass 등
    checked = Column(Boolean, default=False, nullable=False)

    # "HH:MM:SS.cc" 형식 문자열
    start_time = Column(String(20), nullable=False)

    # 최초 bbox 정보 (없을 수도 있어서 nullable=True)
    start_x = Column(Integer, nullable=True)
    start_y = Column(Integer, nullable=True)
    start_w = Column(Integer, nullable=True)
    start_h = Column(Integer, nullable=True)

    clip_name = Column(String(255), nullable=False)
    clip_path = Column(Text, nullable=False)
    thumbnail = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="clips")

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "class_name": self.class_name,
            "checked": self.checked,
            "start_time": self.start_time,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "start_w": self.start_w,
            "start_h": self.start_h,
            "clip_name": self.clip_name,
            "clip_path": self.clip_path,
            "thumbnail": self.thumbnail,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
