from flask import Blueprint, request, jsonify, current_app

user_bp = Blueprint('user', __name__)


class UserRoutes:
    @staticmethod
    @user_bp.route('/login', methods=['POST'])
    def login():
        data = request.get_json()
        print(data)
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({
                "error": "Invalid request. Please provide both username and password."
            }), 400

        username = data["username"]
        password = data["password"]
        user_from_db = current_app.config["USER_SERVICE"].login(username, password)
        if user_from_db:
            return jsonify({
                "current_user_id" : user_from_db.id,
                "current_user_email": user_from_db.email,
                "current_user_name" : user_from_db.uname
            })
        else:
            return jsonify({
                "error": "Invalid username or password or user does not exist"
            }), 401

    @staticmethod
    @user_bp.route('/register', methods=['POST'])
    def register():
        data = request.get_json()
        print(data)
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({
                "error": "Invalid request. Please provide both username and password."
            }), 400
        username = data["username"]
        password = data["password"]
        email_from_client = data["email"]

        current_app.config["USER_SERVICE"].register(username, email_from_client, password)
        user_from_db = current_app.config["USER_SERVICE"].login(username, password)
        if user_from_db:
            return jsonify({
                "current_user_id": user_from_db.id,
                "current_user_email": user_from_db.email,
                "current_user_name": user_from_db.uname
            })
        else:
            return jsonify({
                "error": "Register failed"
            }), 401
