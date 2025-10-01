# Importaciones necesarias
from flask import Flask, request, jsonify, send_from_directory, render_template
import subprocess
import os
import uuid
from flask_cors import CORS

# Configuración de Flask
# Flask buscará el archivo .html en una carpeta llamada 'templates'
app = Flask(__name__, static_folder='uploads', template_folder='templates')
CORS(app) 

# Directorio para guardar archivos temporalmente
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    # Esta ruta servirá tu archivo HTML principal
    return render_template('analizador_termico.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró el archivo de imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    if file:
        unique_id = str(uuid.uuid4())
        filename = unique_id + '_' + os.path.basename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        try:
            # NOTA: Asegúrate de haber modificado tu mainProcessor.py para que
            # la ruta de la imagen la tome de un argumento de línea de comandos.
            print(f"Ejecutando proceso para: {image_path}")
            # Se usa python3 explícitamente para compatibilidad en entornos de despliegue
            subprocess.run(['python3', 'mainProcessor.py', image_path], check=True, capture_output=True, text=True)

            base_name_with_id = os.path.splitext(filename)[0]
            
            face_img_out = f"{base_name_with_id}_rostro_etiquetado.png"
            face_pdf_out = f"{base_name_with_id}_Informe_Rostro.pdf"
            hands_img_out = f"{base_name_with_id}_manos_etiquetada.png"
            hands_pdf_out = f"{base_name_with_id}_Informe_Manos.pdf"
            
            return jsonify({
                'face_image_url': f'/uploads/{face_img_out}',
                'face_pdf_url': f'/uploads/{face_pdf_out}',
                'hands_image_url': f'/uploads/{hands_img_out}',
                'hands_pdf_url': f'/uploads/{hands_pdf_out}'
            })

        except subprocess.CalledProcessError as e:
            print("Error durante la ejecución del script:")
            print("STDERR:", e.stderr)
            return jsonify({'error': 'Error al procesar la imagen', 'details': e.stderr}), 500
        except Exception as e:
            print(f"Error inesperado: {e}")
            return jsonify({'error': str(e)}), 500

# Ruta para servir los archivos generados
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Configuración para despliegue en Railway (o localmente)
    # Railway proporciona la variable de entorno PORT.
    port = int(os.environ.get('PORT', 5000))
    # Escuchar en 0.0.0.0 hace que sea accesible externamente.
    app.run(debug=False, host='0.0.0.0', port=port)
