from flask import Flask, request
from flask_restplus import Api, Resource
from flask_restplus.reqparse import RequestParser
from threading import Thread
from conv import ConversationalModel

# Do flask initialisation
app = Flask(__name__)
api = Api()
api.init_app(app)
# Do model initialisation
model = ConversationalModel()
initializerThread = Thread(
    target=model.InitModel(),
    daemon=False,
    args=[]
    )

@api.route('/ready')
class Ready(Resource):
    def get(self):
        global model
        if model.is_ready:
            return {
                "success": True
            }, 200
        else:
            return {
                "error": "Model isn't ready"
            }, 501

@api.route('/conversation')
class Conversation(Resource):
    def get(self):
        return {"error":
            "Don't GET me. Required fields: {history: string[], query: string}"
        }, 401
    def post(self):
        global model
        if not request.is_json:
            return {"error": "Please transmit data in 'application/json'"}, 400
        data = request.get_json()
        print(data)
        try:
            assert "history" in data and "query" in data
            response = model.Sample(data['history'], data['query'])
            return {'response': response}, 200
        except AssertionError:
            return {"error": "Missing required fields"}, 501
        except Exception: # For catching errors in model.Sample
            return {"error": "A server error occurred"}, 501
        


if __name__ == "__main__":
    app.run(debug=True)
    initializerThread.start()