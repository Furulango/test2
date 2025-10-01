import os
import sys
import subprocess
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import serverless_wsgi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = Flask(__name__, static_folder=os.path.abspath('.'))
CORS(app)

UPLOADS_DIR = '/tmp/uploads'
RESULTS_DIR = '/tmp/results'

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró el archivo de imagen'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    try:
        unique_id = str(uuid.uuid4())
        original_filename, extension = os.path.splitext(file.filename)
        input_filename = f"{unique_id}{extension}"
        input_path = os.path.join(UPLOADS_DIR, input_filename)
        file.save(input_path)

        script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mainProcessor.py')
        
        prototxt_src = os.path.join(os.path.dirname(script_path), 'deploy.prototxt.txt')
        caffemodel_src = os.path.join(os.path.dirname(script_path), 'res10_300x300_ssd_iter_140000.caffemodel')
        
        prototxt_dst = os.path.join(RESULTS_DIR, 'deploy.prototxt.txt')
        caffemodel_dst = os.path.join(RESULTS_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
        
        if os.path.exists(prototxt_src):
            subprocess.run(['cp', prototxt_src, prototxt_dst])
        if os.path.exists(caffemodel_src):
            subprocess.run(['cp', caffemodel_src, caffemodel_dst])


        completed_process = subprocess.run(
            ['python3', script_path, input_path],
            capture_output=True, text=True, cwd=RESULTS_DIR, timeout=25
        )

        if completed_process.returncode != 0:
            print("Error en subprocess:", completed_process.stderr)
            return jsonify({'error': 'Falló el script de procesamiento', 'details': completed_process.stderr}), 500

        base_name_processed = os.path.splitext(os.path.basename(input_filename))[0]
        
        face_pdf = f"{base_name_processed}_Informe_Rostro.pdf"
        hands_pdf = f"{base_name_processed}_Informe_Manos.pdf"
        
        results = {
            'face_pdf_url': f'/.netlify/functions/api/results/{face_pdf}' if os.path.exists(os.path.join(RESULTS_DIR, face_pdf)) else None,
            'hands_pdf_url': f'/.netlify/functions/api/results/{hands_pdf}' if os.path.exists(os.path.join(RESULTS_DIR, hands_pdf)) else None,
            'logs': completed_process.stdout
        }

        return jsonify(results)

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'El procesamiento de la imagen tardó demasiado (más de 25 segundos).'}), 504
    except Exception as e:
        print(f"Error general: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def get_result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)

def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)

