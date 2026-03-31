import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = r"d:\dataset cv\model_and_data\tomato_disease_model_efficientnetb3.h5"
IMG_SIZE = (224, 224)

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

def load_prediction_model():
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please check the path.")
        
        # Check if the other model exists (fallback)
        fallback = r"d:\dataset cv\model_and_data\tomato_disease_model.h5"
        if os.path.exists(fallback):
            print(f"Fallback model found at {fallback}. Loading it instead.")
            return tf.keras.models.load_model(fallback)
        return None
        
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model

def predict_frame(model, frame):
    # Resize to the input shape that the model expects
    img = cv2.resize(frame, IMG_SIZE)
    
    # OpenCV loads images in BGR, but Keras training uses RGB. Convert it.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch of 1
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Get the highest probability class
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    class_name = TOMATO_CLASSES[class_idx]
    
    return class_name, confidence

def is_leaf_present(frame):
    """
    A lightning-fast 'pre-check' to see if there's actually a plant/leaf in the image.
    It checks if there's a minimum amount of green/yellow-green color in the frame.
    """
    # Convert image to HSV color space (easier to filter colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for leaves (yellow-green to dark green)
    # Hue: 25 to 85, Saturation: >40 (no grays), Value: >40 (no blacks)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask of the green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate what percentage of the image is leaf-colored
    total_pixels = frame.shape[0] * frame.shape[1]
    green_ratio = cv2.countNonZero(mask) / total_pixels
    
    # If more than 2% of the image is green, we say there's a leaf.
    return green_ratio > 0.02

def get_disease_info(class_name):
    if class_name == "Tomato_healthy":
        return "Healthy Tomato Leaf", (0, 255, 0) # Green text (BGR format for OpenCV)
    else:
        disease = class_name.replace("Tomato_", "").replace("_", " ")
        return f"Diseased: {disease}", (0, 0, 255) # Red text

def detect_from_image(model):
    img_path = input("\nEnter the full path to the image file: ").strip()
    
    # Remove quotes if user dragged and dropped the file into the terminal
    if img_path.startswith('"') and img_path.endswith('"'):
        img_path = img_path[1:-1]
    # Handle single quotes as well
    if img_path.startswith("'") and img_path.endswith("'"):
        img_path = img_path[1:-1]
        
    if not os.path.exists(img_path):
        print("Image not found! Please check the path.")
        return

    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load image. It might be corrupt or an unsupported format.")
        return

    # PRE-CHECK: Is there a leaf in the image?
    if not is_leaf_present(frame):
        label_text = "No Leaf Detected!"
        color = (0, 165, 255) # Orange for warning
        confidence = 0.0
        class_name = "N/A"
        print("\n--- Detection Result ---")
        print("Status: No Plant/Leaf found in image.")
        print("------------------------\n")
    else:
        # Only run the heavy AI model if a leaf is present
        class_name, confidence = predict_frame(model, frame)
        label_text, color = get_disease_info(class_name)
        
        print(f"\n--- Detection Result ---")
        print(f"Status: {label_text}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw Class: {class_name}")
        print("------------------------\n")
    
    # Display the image with the prediction
    cv2.putText(frame, f"{label_text} ({confidence:.1f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    # Create a resizable window to fit the screen
    cv2.namedWindow("Tomato Leaf Detection - Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Tomato Leaf Detection - Image", frame)
    print("Image window opened. Press any key ON THE IMAGE WINDOW to close it...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_live(model):
    print("\nStarting live camera feed... Press 'q' ON THE CAMERA WINDOW to quit.")
    # Initialize the camera (0 is usually the default built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
            
        # PRE-CHECK: Is there a leaf in the camera?
        if not is_leaf_present(frame):
            label_text = "Searching for Leaf..."
            color = (0, 165, 255) # Orange
            confidence_text = ""
        else:
            class_name, confidence = predict_frame(model, frame)
            label_text, color = get_disease_info(class_name)
            confidence_text = f" ({confidence:.1f}%)"
        
        # Display the result on the frame
        cv2.putText(frame, f"{label_text}{confidence_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    
        cv2.imshow('Tomato Leaf Detection - Live', frame)
        
        # Wait 1ms and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=========================================")
    print("   Tomato Leaf Disease Detector 🍅       ")
    print("=========================================")
    
    model = load_prediction_model()
    if model is None:
        return
        
    while True:
        print("\nSelect an option:")
        print("1. Detect from Image Path")
        print("2. Detect from Live Camera")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            detect_from_image(model)
        elif choice == '2':
            detect_live(model)
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
