# server/auth/register.py
from flask import Blueprint, request, jsonify
from models import db, User

register_bp = Blueprint("register", __name__)


@register_bp.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    password_check = data.get("passwordCheck")

    if not username or not password or not password_check:
        return jsonify({"error": "Missing fields"}), 400

    if password != password_check:
        return jsonify({"error": "Passwords do not match"}), 400

    # 중복 사용자 확인
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409

    # 사용자 저장
    new_user = User(username=username, password=password, video="")
    db.session.add(new_user)
    db.session.commit()

    return (
        jsonify(
            {
                "message": f"{username} signed up successfully!",
                "user": new_user.to_dict(),
            }
        ),
        201,
    )
