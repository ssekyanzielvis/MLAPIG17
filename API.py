from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit (Render has memory limits)

# Custom Lambda layer with output shape specification
def contrastive_loss_layer(embeddings):
    return tf.abs(embeddings[0] - embeddings[1])

# Load the trained model with custom objects
def load_model():
    try:
        model = tf.keras.models.load_model(
            'ecg_psychological_model.h5',
            custom_objects={'contrastive_loss_layer': Lambda(contrastive_loss_layer, output_shape=(1,))}
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'txt'}

@app.route('/test')
def test():
    return "Flask is running!"

@app.route('/')
def index():
    return render_template('index.html', now=datetime.now())

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded", now=datetime.now())
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file", now=datetime.now())
    
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Only CSV or TXT allowed.", now=datetime.now())

    try:
        # Read file directly from memory
        file_content = file.stream.read().decode('utf-8')
        file_stream = io.StringIO(file_content)
        
        # Load and preprocess ECG data
        try:
            ecg_data = pd.read_csv(file_stream)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return render_template('index.html', error=f"Error reading file: {str(e)}", now=datetime.now())
        
        # Validate data
        if ecg_data.empty or len(ecg_data) < 100:
            return render_template('index.html', 
                                 error="Insufficient ECG data (minimum 100 samples required)", 
                                 now=datetime.now())
        
        # Column name normalization
        column_map = {
            'LeadII': 'MLII', 'leadII': 'MLII', 'mlii': 'MLII',
            'LeadV5': 'V5', 'leadV5': 'V5', 'v5': 'V5',
            'LeadV1': 'V1', 'leadV1': 'V1', 'v1': 'V1',
            'LeadV2': 'V2', 'leadV2': 'V2', 'v2': 'V2'
        }
        ecg_data.rename(columns=column_map, inplace=True)
        
        # Determine which pair to use
        leads_used = None
        for pair in [('MLII', 'V5'), ('MLII', 'V1'), ('MLII', 'V2'), ('V5', 'V2')]:
            if all(col in ecg_data.columns for col in pair):
                ecg_samples = ecg_data[list(pair)].values
                leads_used = " and ".join(pair)
                break
        
        if not leads_used:
            missing = [col for col in ['MLII', 'V5', 'V1', 'V2'] 
                      if col not in ecg_data.columns]
            return render_template('index.html', 
                                 error=f"Missing required ECG leads. Need one of: MLII+V5, MLII+V1, MLII+V2, or V5+V2. Missing: {', '.join(missing)}", 
                                 now=datetime.now())
        
        # Generate predictions
        if model:
            reference_sample = ecg_samples[0:1]
            input_pairs = np.array([(reference_sample, sample) for sample in ecg_samples[1:100]])
            predictions = model.predict([input_pairs[:,0], input_pairs[:,1]])
            stress_level = np.mean(predictions)
            anxiety_level = np.median(predictions)
            depression_level = np.max(predictions)
        else:
            return render_template('index.html', 
                                 error="Model failed to load", 
                                 now=datetime.now())
        
        # Generate ECG plot
        ecg_plot = generate_ecg_plot(ecg_samples[:200, 0])
        
        return render_template('index.html', 
                            stress=classify_level(stress_level),
                            anxiety=classify_level(anxiety_level),
                            depression=classify_level(depression_level),
                            ecg_plot=ecg_plot,
                            filename=file.filename,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            now=datetime.now(),
                            leads_used=leads_used)
    
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return render_template('index.html', 
                             error=f"Processing error: {str(e)}", 
                             now=datetime.now())

def classify_level(value):
    """Classify psychological state level based on value"""
    value = float(value)
    if value > 0.7:
        return {'level': 'High', 'value': f"{value:.2f}", 'color': 'danger'}
    elif value > 0.4:
        return {'level': 'Moderate', 'value': f"{value:.2f}", 'color': 'warning'}
    else:
        return {'level': 'Low', 'value': f"{value:.2f}", 'color': 'success'}

def generate_ecg_plot(samples):
    """Generate ECG plot as base64 encoded image"""
    plt.figure(figsize=(10, 4), facecolor='#f8f9fa')
    plt.plot(samples, color='#007bff', linewidth=1.5)
    plt.title('ECG Signal', pad=20)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)