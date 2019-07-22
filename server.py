# std lib
from threading import Thread

# deps
from flask import Flask, request
from flask_restplus import Api, Resource
from loguru import logger

# app imports
from conv import ConversationalModel

# Do flask initialisation
app = Flask(__name__)
api = Api()
api.init_app(app)
# Do model initialisation
model = ConversationalModel()
model.InitModel()


@api.route('/ready')
class Ready(Resource):
    def get(self):
        if model.is_ready:
            return {"success": True}, 200
        else:
            return {"error": "Model isn't ready"}, 501


@api.route('/conversation')
class Conversation(Resource):
    def get(self):
        return {
            "error":
            "Don't GET me. Required fields: {history: string[], query: string}"
        }, 401

    def post(self):
        if not request.is_json:
            return {"error": "Please transmit data in 'application/json'"}, 400
        data = request.get_json()
        logger.debug(data)
        try:
            assert "history" in data and "query" in data
            response = model.Sample(data['history'], data['query'])
            return {'response': response}, 200
        except AssertionError:
            return {"error": "Missing required fields"}, 501
        except Exception:  # For catching errors in model.Sample
            return {"error": "A server error occurred"}, 501
