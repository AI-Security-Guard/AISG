import os
from flask import Blueprint, request, jsonify
from models import User
from flask import send_file
from flask_jwt_extended import jwt_required, get_jwt_identity


bring_video_bp = Blueprint("bringVideo", __name__)
UPLOAD_FOLDER = "uploaded_videos"


@bring_video_bp.route("/bringVideo", methods=["GET"])
@jwt_required()
def bring_video():
    username = request.args.get("username")

    if not username:
        return jsonify({"error": "username이 없음"}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "유저가 없음"}), 404

    # 영상이 없는 경우: 파일 경로 자체가 없거나, 빈 값
    if not user.video or not os.path.exists(user.video):
        return (
            jsonify({"message": "해당 유저는 영상이 없습니다.", "hasVideo": False}),
            200,
        )

    return send_file(user.video, as_attachment=True)
