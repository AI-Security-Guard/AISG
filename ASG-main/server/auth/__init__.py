# server/auth/__init__.py
from .register import register_bp
from .login import login_bp


def register_auth_blueprints(app):
    app.register_blueprint(register_bp)
    app.register_blueprint(login_bp)
