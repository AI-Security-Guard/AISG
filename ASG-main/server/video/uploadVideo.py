import os
from flask import Blueprint, request, jsonify
from models import db, User
from flask_jwt_extended import jwt_required, get_jwt_identity


upload_video_bp = Blueprint("uploadVideo", __name__)
UPLOAD_FOLDER = "uploaded_videos"


@upload_video_bp.route("/uploadVideo", methods=["POST"])
@jwt_required()
def upload_video():
    username = get_jwt_identity()
    file = request.files.get("video")

    if not username or not file:
        return jsonify({"error": "Missing fields"}), 400

    user = User.query.filter_by(username=username).first()

    if not user:
        return jsonify({"error": "User not found"}), 404

    # 이미 video가 존재하면 업로드 거부
    if user.video:
        return jsonify({"error": "User already has a video"}), 400

    # 폴더 생성 후 저장
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = f"{username}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # DB에 경로 저장
    user.video = save_path
    if not user.original_video:
        user.original_video = save_path
    db.session.commit()

    return (
        jsonify(
            {
                "message": "Video uploaded successfully",
                "user": {
                    **user.to_dict(),
                    "full_path": os.path.abspath(save_path),
                },
            }
        ),
        200,
    )
