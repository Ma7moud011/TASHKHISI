from flask import Flask, request, jsonify ,render_template, flash, url_for,session, redirect,send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests, json, hashlib, hmac, base64
from datetime import timedelta, datetime
import numpy as np, pandas as pd
import joblib, os, cv2
from ultralytics import YOLO
from fastai.vision.all import *
from PIL import Image as PILImage
from fastai.vision.all import PILImage

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = '1234'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=90)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/test'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(150), nullable=False, unique=False)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(300), nullable=False)

class Symptoms(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)

class Issues(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symptoms = db.Column(db.JSON, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    birth_year = db.Column(db.Integer, nullable=False)
    diagnosis = db.Column(db.JSON, nullable=False)
    diagnosis_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', backref=db.backref('history', lazy=True))



AUTH_URL = "https://authservice.priaid.ch/login"
SECRET_KEY = "c5K8NkSg47Bsp2MDq"
API_KEY = "c8G9K_GMAIL_COM_AUT"


cached_token = None
token_expiry_time = None

def create_hmac_hash(uri, secret_key):

    hmac_md5 = hmac.new(secret_key.encode(), uri.encode(), hashlib.md5).digest()
    return base64.b64encode(hmac_md5).decode()

def api_login():

    global cached_token, token_expiry_time
    current_time = datetime.now()

    if cached_token and token_expiry_time and current_time < token_expiry_time:

        print("Using cached token (valid until:", token_expiry_time, ")")
        return cached_token


    print("Requesting new token from Priaid API")

    uri = AUTH_URL


    hashed_credentials = create_hmac_hash(uri, SECRET_KEY)

    headers = {
        'Authorization': f'Bearer {API_KEY}:{hashed_credentials}'
    }


    try:
        response = requests.post(uri, headers=headers)
        response.raise_for_status()

        token_data = response.json()


        cached_token = token_data["Token"]
        valid_seconds = token_data["ValidThrough"] - 60
        token_expiry_time = current_time + timedelta(seconds=valid_seconds)
        print("New token received, valid until:", token_expiry_time)
        return cached_token
    except requests.exceptions.RequestException as e:
            print(f"Error obtaining token: {e}")
            raise Exception(f"Failed to authenticate with Priaid API: {e}")

def priaid_api_request(url, params, method='GET', data=None, debug=False):


    token = api_login()
    params['token'] = token


    if debug:
        print(f"Calling Priaid API: {url}")
        print(f"Parameters: {params}")

    try:

        if method.upper() == 'GET':
            response = requests.get(url, params=params)
        else:
            response = requests.post(url, params=params, json=data)

        if debug:
            print(f"API response status: {response.status_code}")


        if response.status_code == 401:
            if debug:
                print("Received 401 Unauthorized, refreshing token and retrying...")


            global cached_token, token_expiry_time
            cached_token = None
            token_expiry_time = None


            token = api_login()
            params['token'] = token


            if method.upper() == 'GET':
                response = requests.get(url, params=params)
            else:
                response = requests.post(url, params=params, json=data)

            if debug:
                print(f"Retry API response status: {response.status_code}")


        if 'application/json' in response.headers.get('Content-Type', ''):
            response_data = response.json()
        else:
            response_data = response.text

        if debug and response.status_code == 200:
            print(f"API response data type: {type(response_data)}")

        return response_data, response.status_code

    except Exception as e:
        error_message = f"Error in Priaid API request to {url}: {str(e)}"
        print(error_message)
        raise Exception(error_message)

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    symptoms = Symptoms.query.all()
    return jsonify([{'id': symptom.id, 'name': symptom.name} for symptom in symptoms])

@app.route('/api/diagnose', methods=['POST'] )
def receive_data():

    data = request.get_json()

    symptom_ids = data['symptom_ids']
    symptom_names = data['symptom_names']
    gender = data['gender']
    year_of_birth = data['birth_year']
    language = "en-gb"

    url = "https://healthservice.priaid.ch/diagnosis"

    try:

        token = api_login()
        params = {
            'token': token,
            'symptoms': json.dumps(symptom_ids),
            'gender': gender,
            'year_of_birth': year_of_birth,
            'language': language
        }


        response = requests.get(url, params=params)



        if response.status_code == 401:
            print("Received 401 Unauthorized, refreshing token and retrying...")
            global cached_token, token_expiry_time
            cached_token = None
            token_expiry_time = None


            token = api_login()
            params['token'] = token
            response = requests.get(url, params=params)


        response.raise_for_status()


        diagnosis_result = response.json()
        if diagnosis_result and 'user_id' in session:
            history_entry = History(
                user_id=session['user_id'],
                symptoms=json.dumps(symptom_names),
                gender=gender,
                birth_year=year_of_birth,
                diagnosis=json.dumps(diagnosis_result)
            )
            db.session.add(history_entry)
            db.session.commit()

        return jsonify(diagnosis_result), response.status_code

    except Exception as e:
        print(f"Error in diagnosis API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/illnesses', methods=['GET'])
def get_illnesses():

    issues = Issues.query.all()


    return jsonify([{'id': issue.id, 'name': issue.name} for issue in issues])

@app.route('/api/illness-info/<int:issue_id>', methods=['GET'])
def get_illness_info(issue_id):

    url = f"https://healthservice.priaid.ch/issues/{issue_id}/info"

    try:

        token = api_login()
        params = {
            'token': token,
            'issue_id': issue_id,
            'language': 'en-gb'
        }


        response = requests.get(url, params=params)



        if response.status_code == 401:
            print("Received 401 Unauthorized, refreshing token and retrying...")
            global cached_token, token_expiry_time
            cached_token = None
            token_expiry_time = None


            token = api_login()
            params['token'] = token
            response = requests.get(url, params=params)


        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Failed to fetch data from external API: {response.text}"}), response.status_code

    except Exception as e:
        print(f"Error in illness-info API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/proposed-symptoms', methods=['POST'])
def get_proposed_symptoms():
    data = request.get_json()

    symptom_ids = data['symptom_ids']
    gender = data['gender']
    year_of_birth = data['birth_year']
    language = "en-gb"

    url = "https://healthservice.priaid.ch/symptoms/proposed"

    try:

        token = api_login()
        params = {
            'token': token,
            'symptoms': json.dumps(symptom_ids),
            'gender': gender,
            'year_of_birth': year_of_birth,
            'language': language
        }


        response = requests.get(url, params=params)



        if response.status_code == 401:
            print("Received 401 Unauthorized, refreshing token and retrying...")
            global cached_token, token_expiry_time
            cached_token = None
            token_expiry_time = None


            token = api_login()
            params['token'] = token
            response = requests.get(url, params=params)


        return jsonify(response.json()), response.status_code

    except Exception as e:
        print(f"Error in proposed-symptoms API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/body-locations', methods=['GET'])
def get_body_locations():
    language = "en-gb"

    url = "https://healthservice.priaid.ch/body/locations"

    try:

        params = {'language': language}
        response_data, status_code = priaid_api_request(
            url=url,
            params=params,
            method='GET'
        )

        return jsonify(response_data), status_code

    except Exception as e:
        print(f"Error in body-locations API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/body/locations/<int:location_id>/sublocations', methods=['GET'])
def get_body_sublocations(location_id):
    language = "en-gb"


    url = "https://healthservice.priaid.ch/body/locations/{}".format(location_id)

    try:

        params = {'language': language}
        print(f"Calling sublocations API: {url}")
        response_data, status_code = priaid_api_request(
            url=url,
            params=params,
            method='GET',
            debug=True
        )

        if status_code == 200:
            print(f"Sublocations API response data: {response_data}")
            return jsonify(response_data), status_code
        else:
            error_message = f"Failed to fetch body sublocations: {response_data}"
            print(error_message)
            return jsonify({"error": error_message}), status_code

    except Exception as e:
        error_message = f"Request error in sublocations: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/api/body-sublocation-symptoms', methods=['POST'])
def get_sublocation_symptoms():
    data = request.get_json()

    sublocation_id = data['sublocation_id']
    gender = data['gender']
    language = "en-gb"


    url = "https://healthservice.priaid.ch/symptoms/{}/{}".format(
        sublocation_id,
        'man' if gender == 'male' else 'woman'
    )

    try:

        params = {'language': language}
        response_data, status_code = priaid_api_request(
            url=url,
            params=params,
            method='GET',
            debug=True
        )

        if status_code == 200:
            return jsonify(response_data), status_code
        else:
            error_message = f"Failed to fetch sublocation symptoms: {response_data}"
            print(error_message)
            return jsonify({"error": error_message}), status_code

    except Exception as e:
        error_message = f"Request error: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/api/symptom-redflag/<int:symptom_id>', methods=['GET'])
def get_symptom_redflag(symptom_id):
    language = "en-gb"

    url = "https://healthservice.priaid.ch/redflag"

    try:

        token = api_login()
        params = {
            'token': token,
            'language': language,
            'symptomId': symptom_id
        }


        print(f"Calling redflag API: {url} with params: {params}")
        response = requests.get(url, params=params)
        print(f"Redflag API response status: {response.status_code}")



        if response.status_code == 401:
            print("Received 401 Unauthorized, refreshing token and retrying...")
            global cached_token, token_expiry_time
            cached_token = None
            token_expiry_time = None


            token = api_login()
            params['token'] = token
            response = requests.get(url, params=params)
            print(f"Retry redflag API response status: {response.status_code}")


        if response.status_code == 200:

            redflag_text = response.text.strip()
            print(f"Redflag API response text: '{redflag_text}'")


            if not redflag_text or redflag_text == '""' or redflag_text == '"' or redflag_text == '"{}' or redflag_text == '{}':
                return jsonify({"redflag": ""}), response.status_code


            if redflag_text.startswith('"') and redflag_text.endswith('"'):
                redflag_text = redflag_text[1:-1]

            return jsonify({"redflag": redflag_text}), response.status_code
        else:
            error_message = f"Failed to fetch red flag: {response.text}"
            print(error_message)
            return jsonify({"error": error_message}), response.status_code

    except Exception as e:
        error_message = f"Request error: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/diagnose')
def diagnose_page():
    return render_template('diagnose.html')

@app.route('/illness guide')
def illness_page():
    return render_template('illness guide.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter(User.email == email).first()
        if existing_user:
            flash('email already exists.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('login'))

    session['form_type'] = 'register'
    return render_template('login&register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user:

            if check_password_hash(user.password, password):

                session['user_id'] = user.id
                session['username'] = user.username
                session.permanent = True
                flash('Login successful!', 'success')
                return redirect(url_for('home_page'))
            else:
                flash('Invalid password.', 'danger')
                return redirect(url_for('login'))
        else:

            flash('Email not found. Please register first.', 'warning')
            return redirect(url_for('register'))

    session['form_type'] = 'login'
    return render_template('login&register.html')

@app.route('/history', methods=['GET'])
def history_page():
    from sqlalchemy import func
    if 'user_id' not in session:
        return redirect('/login')

    user_id = session['user_id']

    all_records = History.query.filter_by(user_id=user_id).order_by(History.diagnosis_date.desc()).all()
    unique_dates = (
        db.session.query(func.date(History.diagnosis_date))
        .filter_by(user_id=user_id)
        .distinct()
        .order_by(func.date(History.diagnosis_date).desc())
        .all()
    )

    date_list = [date[0].strftime('%d-%m-%Y') for date in unique_dates]

    selected_date = request.args.get('filter_date', 'all')

    if selected_date != 'all':
        all_records = [
            record for record in all_records
            if record.diagnosis_date.strftime('%d-%m-%Y') == selected_date
        ]


    for record in all_records:
        record.diagnosis = json.loads(record.diagnosis)
        if isinstance(record.symptoms, str):
            record.symptoms = json.loads(record.symptoms)

    return render_template('history.html', records=all_records, dates=date_list, selected_date=selected_date)

@app.route("/logout")
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home_page'))

@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    if 'user_id' not in session:
        return redirect('/login')


    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect('/login')

    return render_template('profile.html', user=user)

@app.route('/update_username', methods=['POST'])
def update_username():
    if 'user_id' not in session:
        return redirect('/login')
    new_username = request.form['username']
    user = User.query.get(session['user_id'])
    if user:
        if User.query.filter_by(username=new_username).first() and new_username != user.username:
            flash('Username already exists!', 'error')
        else:
            user.username = new_username
            session['username'] = new_username
            db.session.commit()
            flash('Username updated successfully!', 'success')
    return redirect(url_for('profile_page'))

@app.route('/update_password', methods=['POST'])
def update_password():
    if 'user_id' not in session:
        return redirect('/login')
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    user = User.query.get(session['user_id'])
    if user and check_password_hash(user.password, current_password):
        user.password = generate_password_hash(new_password)
        db.session.commit()
        flash('Password updated successfully!', 'success')
    else:
        flash('Current password is incorrect!', 'error')
    return redirect(url_for('profile_page'))

@app.route('/update_email', methods=['POST'])
def update_email():
    if 'user_id' not in session:
        return redirect('/login')
    new_email = request.form['email']
    user = User.query.get(session['user_id'])
    if user:
        if User.query.filter_by(email=new_email).first() and new_email != user.email:
            flash('Email already exists!', 'error')
        else:
            user.email = new_email
            db.session.commit()
            flash('Email updated successfully!', 'success')
    return redirect(url_for('profile_page'))

@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

diabetes_scaler = joblib.load(r"D:\proj\proj\AI models\diabetes_scaler.pkl")
diabetes_pca = joblib.load(r"D:\proj\proj\AI models\diabetes_pca.pkl")
diabetes_model = joblib.load(r"D:\proj\proj\AI models\Diabetes_model_weighted.pkl")

@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    features = ['gender', 'age', 'hypertension', 'heart_disease',
             'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    try:
        data = request.json
        test_data = [data.get(feature, 0) for feature in features]


        df = pd.DataFrame([test_data], columns=features)




        standardized_data = diabetes_scaler.transform(df)
        transformed_data = diabetes_pca.transform(standardized_data)

        prediction = diabetes_model.predict(transformed_data)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/heart')
def heart_page():
    return render_template('heart.html')

heart_scaler = joblib.load(r'D:\proj\proj\AI models\heart_scaler.pkl')
heart_pca = joblib.load(r'D:\proj\proj\AI models\heart_pca.pkl')
heart_model = joblib.load(r'D:\proj\proj\AI models\heart_best_logistic_model.pkl')

@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    data = request.get_json()

    feature_names = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
        'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
        'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    input_data = {
        'Age': float(data['age']),
        'Sex': float(data['sex']),
        'Chest pain type': float(data['cp']),
        'BP': float(data['trestbps']),
        'Cholesterol': float(data['chol']),
        'FBS over 120': float(data['fbs']),
        'EKG results': float(data['restecg']),
        'Max HR': float(data['thalach']),
        'Exercise angina': float(data['exang']),
        'ST depression': float(data['oldpeak']),
        'Slope of ST': float(data['slope']),
        'Number of vessels fluro': float(data['ca']),
        'Thallium': float(data['thal'])
    }


    df = pd.DataFrame([input_data], columns=feature_names)


    scaled_features = heart_scaler.transform(df)
    pca_features = heart_pca.transform(scaled_features)


    prediction = heart_model.predict(pca_features)[0]

    return jsonify({'prediction': int(prediction)})

@app.route('/fetal')
def fetal():
    return render_template('fetal.html')

fetal_scaler = joblib.load(r'D:\proj\proj\AI models\fetal_scaler.pkl')
fetal_model = joblib.load(r'D:\proj\proj\AI models\Fetal_RandomForest.pkl')

@app.route('/fetal_predict', methods=['POST'])
def fetal_predict():
    fetal_FEATURES = [
    'baseline value', 'accelerations', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width',
    'histogram_min', 'histogram_mode', 'histogram_mean', 'histogram_median',
    'histogram_variance', 'histogram_tendency'
]


    fetal_STATUS_MAP = {1: "Normal", 2: "Suspect", 3: "Pathological"}

    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid request. Please send JSON data."}), 400


    missing_features = [feature for feature in fetal_FEATURES if feature not in data]
    if missing_features:
        return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

    try:

        input_df = pd.DataFrame([data], columns=fetal_FEATURES)


        scaled_df = pd.DataFrame(fetal_scaler.transform(input_df), columns=fetal_FEATURES)


        prediction = int(fetal_model.predict(scaled_df)[0])
        return jsonify({"status": fetal_STATUS_MAP.get(prediction, "Unknown")}), 200


    except ValueError:
        return jsonify({"error": "Invalid input values. Please enter numerical values only."}), 400

@app.route('/liver')
def liver_page():
    return render_template('liver.html')

liver_scaler = joblib.load(r'D:\proj\proj\AI models\liver_Robust_scaler.pkl')
liver_model = joblib.load(r'D:\proj\proj\AI models\liver_best_model_1.pkl')

@app.route('/liver_predict', methods=['POST'])
def liver_predict():
    liver_feature_names = ['age', 'gender', 'total_bilirubin',
                  'alkaline_phosphotase', 'alamine_aminotransferase', 'albumin_and_globulin_ratio']

    data = request.get_json()


    input_data = {
        'age': [float(data['age'])],
        'gender': [float(data['gender'])],
        'total_bilirubin': [float(data['total_bilirubin'])],
        'alkaline_phosphotase': [float(data['alkaline_phosphotase'])],
        'alamine_aminotransferase': [float(data['alamine_aminotransferase'])],
        'albumin_and_globulin_ratio': [float(data['albumin_and_globulin_ratio'])]
    }
    df = pd.DataFrame(input_data, columns=liver_feature_names)


    skewed_cols = ['total_bilirubin', 'alkaline_phosphotase',
                    'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    for col in skewed_cols:
        df[col] = df[col].apply(np.log1p)


    X = df[liver_feature_names]


    X_scaled = pd.DataFrame(liver_scaler.transform(X), columns=liver_feature_names)


    prediction = liver_model.predict(X_scaled)[0]


    return jsonify({'prediction': int(prediction)})

@app.route('/malaria')
def malaria_page():
    return render_template('malaria.html')

malaria_UPLOAD_FOLDER = 'static/uploads'
malaria_PROCESSED_FOLDER = 'static/processed'
os.makedirs(malaria_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(malaria_PROCESSED_FOLDER, exist_ok=True)
model_yolo = YOLO(r"D:\proj\proj\AI models\malaria_Yolo_cell_detection.engine", task="detect")
model_fastai = load_learner(r"D:\proj\proj\AI models\malaria_model_fastai_classification.pkl")

def malaria_allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(malaria_PROCESSED_FOLDER, filename)

@app.route('/malaria_upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No files found'}), 400

    files = request.files.getlist('files')
    filenames = []

    for file in files:
        if file.filename == '' or not malaria_allowed_file(file.filename):
            continue
        filename = os.path.join(malaria_UPLOAD_FOLDER, file.filename)
        file.save(filename)
        filenames.append(file.filename)

    return jsonify({'success': True, 'filenames': filenames})

@app.route('/malaria_process', methods=['POST'])
def process_images():
    data = request.get_json()
    filenames = data.get('filenames', [])

    if not filenames:
        return jsonify({'error': 'No filenames provided'}), 400

    results_data = []
    summary = {}
    yolo_summary = {}

    for filename in filenames:
        file_path = os.path.join(malaria_UPLOAD_FOLDER, filename)
        image = cv2.imread(file_path)
        results = model_yolo.predict(file_path, conf=0.15)


        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = PILImage.create(crop)
                label, _, probs = model_fastai.predict(crop_pil)
                fastai_conf = round(float(probs.max()), 2)


                yolo_class = model_yolo.names[int(cls)]

                results_data.append({
                    'filename': filename,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'label': label,
                    'fastai_conf': fastai_conf,
                    'yolo_label': yolo_class,
                    'yolo_conf': float(conf)
                })

                summary[label] = summary.get(label, 0) + 1
                yolo_summary[yolo_class] = yolo_summary.get(yolo_class, 0) + 1


        for det in results_data:
            if det['filename'] == filename:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']

                color = (0, 0, 255) if det['label'] == "Parasitized" else (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        processed_path = os.path.join(malaria_PROCESSED_FOLDER, filename)
        cv2.imwrite(processed_path, image)

    return jsonify({
        'detections': results_data,
        'summary': summary,
        'yolo_summary': yolo_summary,
        'processed_image': f'/processed/{filenames[0]}'
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(host = '0.0.0.0', debug=True)


