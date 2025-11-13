import os
from flask import Blueprint, jsonify
from models import db, User
from flask_jwt_extended import jwt_required, get_jwt_identity

delete_video_bp = Blueprint("deleteVideo", __name__)
UPLOAD_FOLDER = "uploaded_videos"


@delete_video_bp.route("/deleteVideo", methods=["DELETE"])
@jwt_required()
def delete_video():
    # ✅ username은 토큰에서 가져오기
    username = get_jwt_identity()

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    if not user.video:
        return jsonify({"error": "No video to delete"}), 400

    video_path = user.video
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception as e:
            return jsonify({"error": f"File deletion failed: {str(e)}"}), 500

    # DB 초기화
    user.video = None
    db.session.commit()

    return jsonify({"message": "Video deleted successfully"}), 200
