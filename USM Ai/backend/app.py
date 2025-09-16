# app.py
import os
import tempfile
import logging
from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sys # Import sys for potentially exiting on critical errors

# --- Configuration ---
# TODO: IMPORTANT! Update this path to the correct location of your model file.
MODEL_PATH = '/Users/akshitupadhyay/mir_datasets/beatles/models/chord_recognition_best.h5' # <<< CHANGE THIS IF NEEDED!

EXPECTED_SR = 22050  # Sample rate expected by the model/feature extraction
CHORD_LABELS = [     # Ensure this matches your model's output classes exactly
    'C', 'Cm', 'C#', 'C#m', 'D', 'Dm', 'D#', 'D#m',
    'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G', 'Gm',
    'G#', 'G#m', 'A', 'Am', 'A#', 'A#m', 'B', 'Bm',
    'N'  # Often represents "No Chord" or silence
]
# Windowing parameters (can be adjusted)
BEATS_PER_BAR = 4
BARS_PER_WINDOW = 4

# --- Logging Setup ---
# Configure basic logging to show info messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get a logger instance specific to this app
logger = logging.getLogger(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))


template_dir = os.path.join(os.path.dirname(basedir), 'frontend', 'templates')


logger.info(f"Basedir: {basedir}")
logger.info(f"Calculated template directory: {template_dir}")

# Check if the calculated template directory actually exists
if not os.path.isdir(template_dir):
    logger.warning(f"Template directory '{template_dir}' does not exist. "
                   f"Please check your project structure and the template_dir calculation in app.py.")
    # You might want to raise an error here if the frontend is critical
    # raise FileNotFoundError(f"Template directory not found: {template_dir}")

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=template_dir)
# Assign the configured logger to Flask's logger
app.logger = logger

# --- Load Keras Model ---
# Global variable to hold the loaded model
model = None
try:
    if not os.path.exists(MODEL_PATH):
        app.logger.error(f"CRITICAL: Model file not found at specified path: {MODEL_PATH}")
        app.logger.error("Please ensure the MODEL_PATH variable in app.py is correct.")
        # Optionally exit if model is essential for startup:
        # sys.exit("Exiting: Keras model file not found.")
    else:
        model = load_model(MODEL_PATH)
        app.logger.info(f"Keras model loaded successfully from {MODEL_PATH}")
        # Optional: Print model summary to confirm structure
        # model.summary(print_fn=app.logger.info)
except Exception as e:
    app.logger.error(f"CRITICAL: Error loading Keras model from {MODEL_PATH}: {e}", exc_info=True)
    app.logger.error("Prediction endpoint will fail until the model is loaded correctly.")
    # Optionally exit:
    # sys.exit("Exiting: Failed to load Keras model.")

# --- Chord Prediction Logic ---
def predict_chords_and_tempo(audio_file_path):
    """
    Loads audio, predicts chords, timespans, and tempo.

    Args:
        audio_file_path (str): Path to the temporary audio file.

    Returns:
        tuple: (predicted_chords, chord_timespans, song_tempo)
               Returns (None, None, None) if a critical error occurs.
               Types within tuple: (list[str], list[list[float, float]], float)
    """
    if model is None:
         app.logger.error("Model is not loaded. Cannot perform prediction.")
         return None, None, None

    try:
        app.logger.info(f"Processing audio file: {audio_file_path}")
        # Load audio with the expected sample rate
        y, sr = librosa.load(audio_file_path, sr=EXPECTED_SR)
        app.logger.info(f"Audio loaded successfully. Duration: {len(y)/sr:.2f}s, Sample Rate: {sr}")

        # --- Tempo Estimation ---
        app.logger.info("Estimating tempo...")
        # Using aggregate=None can sometimes help if tempo is unstable, but returns an array. Using default.
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time') # Get beat times directly
        # Calculate tempo from beat intervals if beat_track provides times
        if len(beats) > 1:
             estimated_tempo = 60.0 / np.mean(np.diff(beats))
        else:
             # Fallback if few beats detected (or use original tempo output)
             tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr) # Get single tempo value
             estimated_tempo = tempo_val

        # Handle cases where tempo detection might fail or give unrealistic values
        if estimated_tempo is None or not np.isfinite(estimated_tempo) or estimated_tempo <= 30 or estimated_tempo >= 250:
            app.logger.warning(f"Tempo detection returned potentially unstable value ({estimated_tempo}). "
                               f"Consider manual tempo or adjust beat tracking parameters if results are poor. Using fallback tempo.")
            # Try a different aggregate or parameter set as fallback? For now, use a default.
            song_tempo = 120.0 # Default fallback tempo
        else:
            song_tempo = float(estimated_tempo) # Ensure standard Python float

        app.logger.info(f"Estimated tempo: {song_tempo:.2f} BPM")

        # --- Windowing Calculation ---
        # Using beat times directly for windowing can be more robust if beats are accurate
        seconds_per_beat = 60.0 / song_tempo
        bar_duration = seconds_per_beat * BEATS_PER_BAR
        window_duration_seconds = bar_duration * BARS_PER_WINDOW
        window_length_samples = int(window_duration_seconds * sr)

        if window_length_samples <= 0:
             app.logger.error(f"Calculated window length is zero or negative ({window_length_samples} samples). Cannot process.")
             return None, None, None

        num_windows = max(1, int(np.ceil(len(y) / window_length_samples))) # Ensure at least one window
        app.logger.info(f"Window duration: {window_duration_seconds:.2f}s ({window_length_samples} samples) | Estimated windows: {num_windows}")

        # --- Feature Extraction and Prediction per Window ---
        predicted_chords = []
        chord_timespans = []

        for i in range(num_windows):
            start_sample = i * window_length_samples
            end_sample = start_sample + window_length_samples
            # Ensure end_sample doesn't exceed audio length for the last window
            end_sample = min(end_sample, len(y))
            # Calculate actual duration for timespans
            start_time = float(start_sample / sr)
            end_time = float(end_sample / sr)

            # Get the current window segment
            window = y[start_sample:end_sample]

            # Skip if window is effectively empty
            if len(window) < 10: # Arbitrary small number of samples
                app.logger.debug(f"Skipping very short window {i+1} (samples {start_sample}-{end_sample})")
                continue

            app.logger.debug(f"Processing window {i+1}/{num_windows} ({start_time:.2f}s - {end_time:.2f}s)")

            # --- Feature Extraction ---
            # Note: Padding might be needed if the model strictly requires fixed-length input features,
            # even after aggregation. However, aggregation (like mean) handles variable length windows naturally.
            # Padding *before* feature extraction might be better if needed.
            try:
                mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
                chroma = librosa.feature.chroma_stft(y=window, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=window, sr=sr)
                # Using harmonic component for tonnetz is often good but slower
                y_harmonic = librosa.effects.harmonic(window)
                tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            except Exception as feat_err:
                 app.logger.error(f"Error extracting features for window {i+1}: {feat_err}", exc_info=True)
                 # Decide how to handle: Skip window? Assign 'N'? For now, skip.
                 continue # Skip this window if feature extraction fails

            # Combine features: stack vertically (shape: num_features, time_steps_in_window)
            feature_stack = np.vstack([mfcc, chroma, spectral_contrast, tonnetz])
            # Transpose for aggregation (shape: time_steps_in_window, num_features)
            feature_stack_t = feature_stack.T

            # Aggregate features over the window's time steps (e.g., using mean)
            # This assumes your model expects a single feature vector per window chunk.
            if feature_stack_t.shape[0] == 0: # Handle case where feature extraction yields nothing
                app.logger.warning(f"Window {i+1} yielded no features after transpose. Skipping prediction.")
                continue
            # Calculate mean, ignoring potential NaNs if any arise from features
            feature_aggregated = np.nanmean(feature_stack_t, axis=0) # Shape: (num_features,)

            # Check for NaNs after aggregation (if nanmean wasn't used or failed)
            if np.isnan(feature_aggregated).any():
                app.logger.warning(f"Window {i+1} features contain NaNs after aggregation. Skipping prediction.")
                continue

            # --- Reshape for Keras Model Input ---
            # !!! CRITICAL ASSUMPTION !!!
            # The following lines assume your Keras model expects input shape: (batch_size, 7, num_features).
            # The `np.repeat` creates the sequence of length 7 by copying the aggregated features.
            #
            # IF YOUR MODEL EXPECTS A DIFFERENT SHAPE, YOU **MUST** CHANGE THIS:
            # - If model expects (batch_size, num_features) (e.g., Dense input):
            #   feature_reshaped = np.expand_dims(feature_aggregated, axis=0) # Shape: (1, num_features)
            # - If model expects (batch_size, timesteps, num_features) where timesteps is NOT 7:
            #   Adjust the np.repeat count or use a different strategy (e.g., padding features to a fixed length *before* aggregation).
            try:
                num_features = feature_aggregated.shape[0]
                # Reshape to (1, num_features) first
                feature_reshaped_base = np.expand_dims(feature_aggregated, axis=0)
                # Reshape to (1, 1, num_features)
                feature_reshaped_expand = np.expand_dims(feature_reshaped_base, axis=1)
                # Repeat along the time axis (axis 1) 7 times to get (1, 7, num_features)
                feature_reshaped = np.repeat(feature_reshaped_expand, 7, axis=1)
                # ----- END OF CRITICAL SHAPE ASSUMPTION -----

            except Exception as reshape_err:
                app.logger.error(f"Error reshaping features for window {i+1} (Input features: {feature_aggregated.shape}): {reshape_err}", exc_info=True)
                continue # Skip this window

            app.logger.debug(f"Feature shape for model prediction: {feature_reshaped.shape}")

            # --- Predict Chord ---
            try:
                # Use verbose=0 to avoid Keras's own console prediction logs per window
                prediction = model.predict(feature_reshaped, verbose=0)
                # Assuming model output is (batch_size, num_classes) or similar *after processing the sequence*.
                # If the model output is still sequential (e.g., (1, 7, num_classes)), you might need to average or take the last step:
                # Example: prediction_averaged = np.mean(prediction[0], axis=0)
                # Example: prediction_last_step = prediction[0, -1, :]
                # For now, assuming the model reduces the sequence internally or we take the first output:
                prediction_vector = prediction[0] # Shape typically (num_classes,) or (7, num_classes)

                # If prediction is still sequential (e.g., shape (7, 25)), aggregate it:
                if prediction_vector.ndim > 1 and prediction_vector.shape[0] == 7:
                     # Option 1: Mean prediction over the 7 steps
                     # prediction_vector = np.mean(prediction_vector, axis=0)
                     # Option 2: Prediction at the middle step (or last)
                     prediction_vector = prediction_vector[3] # Middle step prediction
                     app.logger.debug("Aggregated sequential prediction output (used middle step).")


                predicted_chord_index = np.argmax(prediction_vector)

                # Map index to chord label
                if 0 <= predicted_chord_index < len(CHORD_LABELS):
                    predicted_chord_label = CHORD_LABELS[predicted_chord_index]
                else:
                    app.logger.warning(f"Predicted index {predicted_chord_index} is out of bounds for CHORD_LABELS (size {len(CHORD_LABELS)}). Using 'N'.")
                    predicted_chord_label = 'N' # Default to No Chord on error

            except Exception as pred_err:
                 app.logger.error(f"Error during model prediction for window {i+1}: {pred_err}", exc_info=True)
                 predicted_chord_label = 'N' # Assign default on prediction error

            predicted_chords.append(predicted_chord_label)
            # Store timespan as [start, end] list of standard floats
            chord_timespans.append([start_time, end_time])
            app.logger.debug(f"Window {i+1}: Predicted '{predicted_chord_label}' ({start_time:.2f}s - {end_time:.2f}s)")


        app.logger.info("Finished processing all windows.")
        # Return lists of strings, lists of floats, and a float
        return predicted_chords, chord_timespans, song_tempo

    except librosa.ParameterError as lib_err:
        app.logger.error(f"Librosa Error processing file {audio_file_path}: {lib_err}", exc_info=True)
        return None, None, None
    except FileNotFoundError:
        # This might happen if the temp file is deleted prematurely elsewhere
        app.logger.error(f"Audio file not found during processing (should be temporary): {audio_file_path}")
        return None, None, None
    except Exception as e:
        # Catch-all for other unexpected errors during processing
        app.logger.error(f"Unexpected error processing audio file {audio_file_path}: {e}", exc_info=True)
        return None, None, None


# --- Flask Routes ---
@app.route('/')
def home():
    """ Home route to serve the index.html page. """
    app.logger.info(f"Accessed home route '/', attempting to render index.html from '{app.template_folder}'")
    try:
        return render_template('index.html')
    except Exception as e:
        # Catch potential jinja exceptions if template is missing/malformed, even if folder exists
        app.logger.error(f"Error rendering template 'index.html': {e}", exc_info=True)
        return "Error rendering frontend page. Check server logs.", 500


@app.route('/predict', methods=['POST'])
def handle_prediction_request():
    """
    Handles POST requests with audio file uploads for chord and tempo prediction.
    Expects a file in the request form data with the key 'audio'.
    """
    app.logger.info("Received POST request to /predict")

    # --- Request Validation ---
    if model is None:
         app.logger.error("Prediction attempt failed: Model was not loaded during startup.")
         return jsonify({'error': 'Server configuration error: Model not loaded.'}), 500 # Internal Server Error

    # Check if the 'audio' file part is present in the request
    if 'audio' not in request.files:
        app.logger.warning("Request rejected: 'audio' file part missing.")
        return jsonify({'error': "Missing 'audio' file part in the request."}), 400 # Bad Request

    uploaded_file = request.files['audio']

    # Check if a file was actually selected and uploaded (filename is not empty)
    if uploaded_file.filename == '':
        app.logger.warning("Request rejected: No file selected for upload.")
        return jsonify({'error': 'No file selected.'}), 400 # Bad Request

    # --- File Handling & Processing ---
    temp_file_path = None  # Initialize path variable outside try block
    try:
        # Create a temporary file to save the upload securely
        # Using NamedTemporaryFile with delete=False requires manual cleanup
        # Get file extension for librosa (though it's often robust)
        _, file_extension = os.path.splitext(uploaded_file.filename)
        # Use a temporary directory managed by Flask/Werkzeug if possible, or system default
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension or '.tmp')
        uploaded_file.save(temp_audio_file)
        temp_file_path = temp_audio_file.name # Get the path for librosa
        temp_audio_file.close() # Close the file handle immediately after saving
        app.logger.info(f"File '{uploaded_file.filename}' saved temporarily to '{temp_file_path}'")

        # --- Call Prediction Logic ---
        predicted_chords, chord_timespans, song_tempo = predict_chords_and_tempo(temp_file_path)

        # --- Check Prediction Results ---
        if predicted_chords is None or chord_timespans is None or song_tempo is None:
             # Errors inside predict_chords_and_tempo should have been logged already
             app.logger.error("Prediction function failed to return valid results.")
             return jsonify({'error': 'Failed to process audio or predict chords due to an internal error.'}), 500

        # --- Prepare Response ---
        # Data should already be in JSON-serializable types
        response_data = {
            'predicted_chords': predicted_chords,
            'chord_timespans': chord_timespans,
            'song_tempo': song_tempo
        }
        app.logger.info("Prediction successful. Returning results.")
        return jsonify(response_data)

    except Exception as e:
        # General error handler for issues during file handling or unexpected issues before/after prediction call
        app.logger.error(f"An unexpected error occurred during prediction request handling: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred processing the request.'}), 500

    finally:
        # --- Cleanup ---
        # Ensure the temporary file is deleted if its path was assigned
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                app.logger.info(f"Temporary file '{temp_file_path}' removed.")
            except OSError as e:
                # Log error but don't crash if cleanup fails
                app.logger.error(f"Error removing temporary file '{temp_file_path}': {e}")

# Add a simple health check endpoint (optional but good practice)
@app.route('/health', methods=['GET'])
def health_check():
    """ Basic health check endpoint """
    status = {'status': 'ok', 'model_loaded': model is not None}
    status_code = 200 if model is not None else 503 # Service Unavailable if model isn't ready
    app.logger.debug(f"Health check accessed. Status: {status}")
    return jsonify(status), status_code

# --- Main Execution ---
if __name__ == '__main__':
    # Run the Flask development server
    # host='0.0.0.0' makes the server accessible from other devices on the network
    # debug=True enables auto-reloading and detailed error pages (DISABLE in production!)
    app.logger.info("Starting Flask development server...")
    # Port 5000 is default, can be changed if needed
    app.run(host='0.0.0.0', port=5001, debug=True)
    # For production, use a proper WSGI server like Gunicorn or Waitress:
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 app:app