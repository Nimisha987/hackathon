# app.py
import os
from dotenv import load_dotenv 
load_dotenv() 
from flask import Flask, render_template, request, redirect, url_for
import os, torch, joblib, requests
from torchvision import models, transforms
from PIL import Image
from transformers import pipeline, set_seed
from werkzeug.utils import secure_filename
from chatbot import get_disease_info 

app = Flask(__name__)

# Define UPLOAD_FOLDER for image uploads.
UPLOAD_FOLDER = 'static/upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

# Set seed for reproducibility (useful for transformers pipelines)
set_seed(42)

# --- Load models ---
# IMPORTANT: These model files (model_1.pth, model.joblib, diabetes_model.joblib,
# scaler_diabetes.joblib) must be present in your project directory for the
# application to run successfully.
try:
    # Image model for skin disease prediction (e.g., ResNet)
    image_model = models.resnet18()
    image_model.fc = torch.nn.Linear(image_model.fc.in_features, 10)
    # Ensure model_1.pth is in the same directory as app.py, or specify full path
    image_model.load_state_dict(torch.load("model_1.pth", map_location='cpu'))
    image_model.eval() # Set model to evaluation mode

    # Text model for symptom classification (e.g., scikit-learn model)
    # Ensure model.joblib is in the same directory as app.py
    text_model = joblib.load("model.joblib")

    # Diabetes prediction models
    # Ensure diabetes_model.joblib and scaler_diabetes.joblib are in the same directory as app.py
    diabetes_model = joblib.load("diabetes_model.joblib")
    scaler = joblib.load("scaler_diabetes.joblib")

    # Summarization pipeline (Hugging Face Transformers)
    # Ensure this model is downloaded or accessible. It downloads on first run.
    summarizer = pipeline("summarization", model="google/pegasus-xsum")

except FileNotFoundError as e:
    print(f"Error loading model: {e}. Please ensure all model files are in the correct directory.")
    exit() 
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()


# Image transformation pipeline for skin disease images
image_transform = transforms.Compose([
    transforms.Resize((224,224)), # Resize images to 224x224 pixels
    transforms.ToTensor() # Convert image to PyTorch tensor
])

# Class names for skin disease prediction
class_names = [
    'viral_infections', 'fungal_infections', 'psoriasis', 'eczema',
    'nevi', 'seborrheic_keratoses', 'bcc', 'atopic_dermatitis',
    'bkl', 'melanoma'
]

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/login')
def login():
    """Renders the login page."""
    return render_template('login.html')

@app.route('/signup')
def signup():
    """Renders the signup page."""
    return render_template('signup.html')


@app.route('/skin', methods=['GET','POST'])
def skin():
    """
    Handles skin disease prediction.
    GET: Displays the skin prediction form.
    POST: Processes uploaded image, performs prediction, and gets chatbot response.
    """
    img_pred = None
    chatbot_response = None

    if request.method == 'POST':
        # Get location from form for chatbot
        location = request.form.get('location', '').strip()
        # Get the uploaded image file
        img = request.files.get('image')

        if img and img.filename:
            # Secure filename to prevent directory traversal attacks
            fn = secure_filename(img.filename)
            # Define the full path to save the image
            path = os.path.join(app.root_path, UPLOAD_FOLDER, fn) 
            img.save(path)

            try:
                # Open and transform the image for model prediction
                image_pil = Image.open(path).convert('RGB')
                tensor = image_transform(image_pil).unsqueeze(0) # Add batch dimension

                with torch.no_grad(): # Disable gradient calculation for inference
                    # Perform prediction using the image model
                    output = image_model(tensor)
                    idx = torch.argmax(output, 1).item() # Get the predicted class index
                    label = class_names[idx] # Get the class name
                    img_pred = label # Store the prediction

                    # Get a detailed chatbot response based on the predicted disease and user location
                    chatbot_response = get_disease_info("", label, location)
            except Exception as e:
                img_pred = f"Error processing image: {e}"
                chatbot_response = "Could not generate detailed info due to image processing error. Please ensure you uploaded a valid image."
            finally:
                # Clean up: optionally remove the uploaded image after processing
                if os.path.exists(path):
                    os.remove(path)
        else:
            img_pred = "No image uploaded."
            chatbot_response = "Please upload an image to get a diagnosis and nearby doctor suggestions."
    
    # Render the skin.html template with prediction and chatbot response
    return render_template('skin.html', img_pred=img_pred, chatbot_response=chatbot_response)

@app.route('/symptom', methods=['GET','POST'])
def symptom():
    """
    Handles symptom classification.
    GET: Displays the symptom classification form.
    POST: Processes symptoms text, performs classification, and gets chatbot response.
    """
    text_pred = None
    chatbot_response = None

    if request.method == 'POST':
        # Get symptoms and location from form
        symp = request.form.get('symptoms', '').strip()
        location = request.form.get('location', '').strip()

        if symp:
            try:
                # Predict the disease label from symptoms text
                label = text_model.predict([symp])[0]
                text_pred = label # Store the prediction

                # Get a detailed chatbot response based on symptoms and user location
                chatbot_response = get_disease_info(symp, "", location)
            except Exception as e:
                text_pred = f"Error classifying symptoms: {e}"
                chatbot_response = "Could not generate detailed info due to symptom classification error. Please try again."
        else:
            text_pred = "Please provide symptoms."
            chatbot_response = "Please enter your symptoms to get a classification and nearby doctor suggestions."
    
    # Render the symptom.html template with prediction and chatbot response
    return render_template('symptom.html', text_pred=text_pred, chatbot_response=chatbot_response)

@app.route('/summary', methods=['GET','POST'])
def summary():
    """
    Handles medical note summarization.
    GET: Displays the summarization form.
    POST: Processes medical notes, generates summary.
    """
    summary_text = None

    if request.method == 'POST':
        note = request.form.get('note', '').strip()

        if not note:
            summary_text = "Please paste medical notes to summarize."
        elif len(note) < 50: # Arbitrary minimum length for meaningful summary
            summary_text = "Please provide longer medical notes for summarization."
        else:
            try:
                # The pipeline handles max_length and min_length for output summary.
                # For very long inputs that might exceed model's context window,
                # you might need to preprocess (e.g., chunking).
                # For now, let's assume the input fits or the pipeline handles internally.
                generated_summary = summarizer(note, max_length=60, min_length=10, do_sample=False)
                summary_text = generated_summary[0]['summary_text']
            except IndexError:
                # This catches the specific "index out of range" error from the summarizer.
                # It often means the model could not produce any output.
                summary_text = "Summarization failed: The provided notes might be in an unsuitable format or too complex/empty for the model to process effectively. Please try different notes."
            except Exception as e:
                # General catch-all for other unexpected errors during summarization
                summary_text = f"An unexpected error occurred during summarization: {e}. Please ensure the notes are in a suitable format."
    
    return render_template('summary.html', summary=summary_text)

@app.route('/diabetes', methods=['GET','POST'])
def diabetes():
    """
    Handles diabetes risk analysis.
    GET: Displays the diabetes risk analysis form (analyzer.html).
    POST: Processes health metrics, performs prediction, and renders the result (diabetes.html).
    """
    result = None

    if request.method == 'POST':
        try:
            # Define the expected fields from the form
            fields = ['pregnancies','glucose','blood_pressure','skin_thickness','insulin','bmi','dpf','age']
            # Convert form data to floats
            feats = [float(request.form[f]) for f in fields]
            
            # Transform features using the loaded scaler and predict
            pred = diabetes_model.predict(scaler.transform([feats]))[0]
            # Determine the result message
            result = "Positive for Diabetes" if pred == 1 else "Negative for Diabetes"
            
            # Render the diabetes.html (result page) with the prediction result
            return render_template('diabetes.html', result=result)
            
        except ValueError:
            # Render analyzer.html with an error message if input is invalid
            error_message = "Error: Please enter valid numerical values for all fields."
            return render_template('analyzer.html', error=error_message)
        except Exception as e:
            # Render analyzer.html with an unexpected error message
            error_message = f"An unexpected error occurred during prediction: {e}"
            return render_template('analyzer.html', error=error_message)
    
    # For GET requests, render the analyzer.html (input form)
    return render_template('analyzer.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
