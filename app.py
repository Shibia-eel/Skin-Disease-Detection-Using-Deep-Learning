from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
import os
import sqlite3

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Two-stage prediction function
from prediction import predict_image


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

# Secret key for login sessions
app.secret_key = "skin_disease_project_secret_key"


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

DATABASE = os.path.join(
    BASE_DIR,
    "users.db"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# DATABASE
# =========================================================

def init_db():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# Initialize database
init_db()


# =========================================================
# HOME PAGE
# =========================================================
# Home page is ALWAYS shown first.
# Login is NOT required to view the home page.

@app.route("/")
def home():

    return render_template(
        "index.html",
        username=session.get("username")
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        # Check empty fields
        if not username or not email or not password:

            return render_template(
                "register.html",
                error="Please fill all fields."
            )

        # Hash password
        hashed_password = generate_password_hash(
            password
        )

        try:

            connection = sqlite3.connect(DATABASE)

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO users
                (username, email, password)
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    email,
                    hashed_password
                )
            )

            connection.commit()
            connection.close()

            # Registration successful
            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            return render_template(
                "register.html",
                error="Username or email already exists."
            )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        connection = sqlite3.connect(
            DATABASE
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT username, password
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

        user = cursor.fetchone()

        connection.close()

        # Check login details
        if user and check_password_hash(
            user[1],
            password
        ):

            # Store username in session
            session["username"] = user[0]

            # After login go directly to prediction page
            return redirect(
                url_for("predict")
            )

        else:

            return render_template(
                "login.html",
                error="Invalid username or password."
            )

    return render_template(
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    # Clear login session
    session.clear()

    # After logout return to HOME page
    return redirect(
        url_for("home")
    )


# =========================================================
# SERVE UPLOADED FILES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================================================
# PREDICTION PAGE + PREDICTION PROCESS
# =========================================================
#
# GET  -> Shows prediction/upload page
# POST -> Processes uploaded image

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Prediction requires login
    if "username" not in session:

        return redirect(
            url_for("login")
        )

    # -----------------------------------------------------
    # GET REQUEST
    # -----------------------------------------------------
    # When user opens /predict directly,
    # show the prediction page.

    if request.method == "GET":

        return render_template(
            "predict.html",
            username=session.get("username")
        )

    # -----------------------------------------------------
    # POST REQUEST
    # -----------------------------------------------------

    image = request.files.get(
        "image"
    )

    # Check whether image was selected
    if image is None or image.filename == "":

        return render_template(
            "predict.html",
            username=session.get("username"),
            error="Please select an image."
        )

    # Secure filename
    filename = secure_filename(
        image.filename
    )

    # Create image path
    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save uploaded image
    image.save(
        image_path
    )

    # =====================================================
    # TWO-STAGE PREDICTION
    # =====================================================

    result = predict_image(
        image_path
    )

    # =====================================================
    # STAGE 1 -> NORMAL SKIN
    # =====================================================

    if result["status"] == "normal":

        return render_template(
            "result.html",

            status="NORMAL",

            disease="Normal Skin",

            confidence=result["confidence"],

            # For normal skin, Stage 1 confidence
            # is also the main confidence
            stage1_confidence=result["confidence"],

            image_filename=filename,

            username=session.get("username")
        )

    # =====================================================
    # STAGE 2 -> DISEASE DETECTED
    # =====================================================

    return render_template(
        "result.html",

        status="DISEASED",

        disease=result["class"],

        confidence=result["confidence"],

        stage1_confidence=result["stage1_confidence"],

        image_filename=filename,

        username=session.get("username")
    )


# =========================================================
# TREATMENT / MEDICINAL PLANT RECOMMENDATION PAGE
# =========================================================
#
# Example:
# /treatment/MEL
# /treatment/AK
# /treatment/BCC
#
# This page will be opened from result.html.

@app.route("/treatment/<disease>")
def treatment(disease):

    # Treatment recommendation requires login
    if "username" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "treatment.html",

        disease=disease,

        username=session.get("username")
    )


# =========================================================
# PROFILE PAGE
# =========================================================

@app.route("/profile")
def profile():

    # Profile requires login
    if "username" not in session:

        return redirect(
            url_for("login")
        )

    username = session.get(
        "username"
    )

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT username, email
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    user = cursor.fetchone()

    connection.close()

    if user:

        return render_template(
            "profile.html",
            username=user[0],
            email=user[1]
        )

    return redirect(
        url_for("login")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )