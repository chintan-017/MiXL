"""
MiXL Flask Server — Intelligent DJ Edition
"""
import os
import json
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

UPLOAD_DIR = Path('static/uploads')
OUTPUT_DIR = Path('static/output')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}


def allowed(filename):
    return Path(filename).suffix.lower() in {'.wav'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename or not allowed(f.filename):
        return jsonify({'error': 'Only WAV files supported'}), 400

    fid = str(uuid.uuid4())[:8]
    fname = f"{fid}.wav"
    fpath = UPLOAD_DIR / fname
    f.save(str(fpath))

    try:
        from audio_engine import analyze_track
        info = analyze_track(str(fpath))
        info['file_id'] = fid
        info['url'] = f'/static/uploads/{fname}'
        info.pop('energy_times', None)
        info.pop('energy_rms', None)
        return jsonify({'success': True, 'track': info})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/mix', methods=['POST'])
def mix():
    data = request.json
    fid1 = data.get('file_id_1')
    fid2 = data.get('file_id_2')

    matches1 = list(UPLOAD_DIR.glob(f"{fid1}.*"))
    matches2 = list(UPLOAD_DIR.glob(f"{fid2}.*"))

    if not matches1 or not matches2:
        return jsonify({'error': 'One or more files not found'}), 404

    job_id = str(uuid.uuid4())[:8]
    out_name = f"mix_{job_id}.wav"
    out_path = str(OUTPUT_DIR / out_name)

    jobs[job_id] = {'status': 'processing', 'progress': 'Initializing AI DJ...', 'log': []}

    def run_mix():
        try:
            from dj_brain import intelligent_mix

            def progress_cb(msg):
                jobs[job_id]['progress'] = msg
                jobs[job_id]['log'].append(msg)

            result = intelligent_mix(
                str(matches1[0]),
                str(matches2[0]),
                out_path,
                progress_cb=progress_cb
            )

            # Clean non-serializable fields
            for tk in ['track1', 'track2']:
                result[tk].pop('beats', None)
                result[tk].pop('energy_times', None)
                result[tk].pop('energy_rms', None)
                result[tk].pop('filepath', None)

            result['output_url'] = f'/static/output/{out_name}'
            existing_log = jobs[job_id].get('log', [])
            jobs[job_id] = {
                'status': 'done',
                'result': result,
                'output_url': f'/static/output/{out_name}',
                'log': existing_log,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            jobs[job_id] = {'status': 'error', 'error': str(e)}

    t = threading.Thread(target=run_mix)
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id, {'status': 'not_found'})
    return jsonify(job)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    print("🎵 MiXL Intelligent AI DJ — http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)