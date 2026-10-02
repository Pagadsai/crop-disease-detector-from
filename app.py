from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

app = FastAPI(title="Crop Disease Detector")


MODEL_PATH = "model/best_mobilenet.tflite"

CLASSES = [
    "bell_pepper_healthy",
    "bell_pepper_leaf_spot",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl"
]

PLANT_INDICES = {
    "Bell Pepper": [0, 1],
    "Potato": [2, 3, 4],
    "Tomato": [5, 6, 7, 8, 9, 10, 11, 12]
}

NAMES = {
    "bell_pepper_healthy": "Healthy",
    "bell_pepper_leaf_spot": "Leaf Spot",
    "potato_early_blight": "Early Blight",
    "potato_healthy": "Healthy",
    "potato_late_blight": "Late Blight",
    "tomato_bacterial_spot": "Bacterial Spot",
    "tomato_early_blight": "Early Blight",
    "tomato_healthy": "Healthy",
    "tomato_late_blight": "Late Blight",
    "tomato_leaf_mold": "Leaf Mold",
    "tomato_mosaic_virus": "Mosaic Virus",
    "tomato_septoria_leaf_spot": "Septoria Leaf Spot",
    "tomato_yellow_leaf_curl": "Yellow Leaf Curl Virus"
}

TREATMENT = {
    "Healthy": (
        "The leaf appears healthy. Continue regular watering, "
        "proper nutrition and monitor the plant regularly."
    ),
    "Early Blight": (
        "Remove affected leaves, improve air circulation and avoid "
        "overhead watering. Consider an appropriate fungicide if needed."
    ),
    "Late Blight": (
        "Remove infected leaves, keep foliage dry and improve airflow. "
        "Consider an appropriate fungicide and monitor nearby plants."
    ),
    "Bacterial Spot": (
        "Remove affected leaves and avoid overhead watering. "
        "Keep foliage dry and maintain good plant spacing."
    ),
    "Leaf Mold": (
        "Improve ventilation and reduce humidity around the plant. "
        "Remove severely affected leaves."
    ),
    "Mosaic Virus": (
        "Remove severely infected plants and control insect vectors "
        "such as aphids. Avoid handling healthy plants after infected ones."
    ),
    "Septoria Leaf Spot": (
        "Remove affected leaves, improve airflow and avoid wetting "
        "the foliage when watering."
    ),
    "Yellow Leaf Curl Virus": (
        "Control whiteflies and remove severely infected plants. "
        "Monitor nearby plants for symptoms."
    ),
    "Leaf Spot": (
        "Remove affected leaves and improve air circulation. "
        "Avoid prolonged moisture on the foliage."
    )
}


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()




def predict(image, selected_plant):
    
    image = image.convert("RGB")
    image = image.resize((224, 224))

    arr = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = ((arr - mean) / std).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(
        input_details[0]["index"],
        arr
    )

    interpreter.invoke()

    logits = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Softmax
    exp = np.exp(logits - np.max(logits))
    probabilities = exp / np.sum(exp)

    # Find the model's actual prediction
    best_index = int(np.argmax(probabilities))

    predicted_class = CLASSES[best_index]
    predicted_confidence = float(
        probabilities[best_index] * 100
    )

    # Determine actual plant
    if predicted_class.startswith("tomato_"):
        predicted_plant = "Tomato"

    elif predicted_class.startswith("potato_"):
        predicted_plant = "Potato"

    elif predicted_class.startswith("bell_pepper_"):
        predicted_plant = "Bell Pepper"

    else:
        predicted_plant = "Unknown"

    # Prevent wrong crop selection
    if predicted_plant != selected_plant:

        return {
            "mismatch": True,
            "predicted_plant": predicted_plant,
            "confidence": round(predicted_confidence, 2),
            "message": (
                f"The image appears to be a "
                f"{predicted_plant} leaf. "
                f"Please select {predicted_plant} and analyze again."
            )
        }

    display_name = NAMES[predicted_class]

    return {
        "mismatch": False,
        "plant": predicted_plant,
        "disease": display_name,
        "confidence": round(predicted_confidence, 2),
        "treatment": TREATMENT.get(
            display_name,
            "Consult an agricultural expert for further diagnosis."
        )
    }


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)



@app.get("/", response_class=HTMLResponse)
async def home():

    with open("templates/index.html", "r") as file:
        return file.read()



@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    plant: str = Form(...)
):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        return {
            "error": "Please upload a JPG, PNG or WebP image."
        }

    if plant not in PLANT_INDICES:
        return {
            "error": "Please select a valid plant."
        }

    try:

        data = await file.read()

        if len(data) > 10 * 1024 * 1024:
            return {
                "error": "Image must be smaller than 10 MB."
            }

        image = Image.open(io.BytesIO(data))

        result = predict(image, plant)

        return result

    except Exception as e:

        return {
            "error": f"Unable to analyze this image: {str(e)}"
        }