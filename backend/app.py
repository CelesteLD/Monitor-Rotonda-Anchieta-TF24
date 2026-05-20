import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from routes.status  import status_bp
from routes.frame   import frame_bp
from routes.history import history_bp
from routes.events  import events_bp
from routes.spark  import spark_bp

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.register_blueprint(status_bp,  url_prefix='/api')
app.register_blueprint(frame_bp,   url_prefix='/api')
app.register_blueprint(history_bp, url_prefix='/api')
app.register_blueprint(events_bp,  url_prefix='/api')
app.register_blueprint(spark_bp,   url_prefix='/api')

@app.route('/api/health')
def health():
    return {'status': 'ok'}

_scheduler_started = False

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    from services.scheduler import start_scheduler
    start_scheduler()

if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    _start_scheduler_once()

if not app.debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    _start_scheduler_once()

if __name__ == '__main__':
    _start_scheduler_once()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)