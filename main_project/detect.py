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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Base image
    img_array1 = tf.keras.preprocessing.image.img_to_array(img_rgb)
    
    # 2. Horizontal Flip
    img_flip = cv2.flip(img_rgb, 1)
    img_array2 = tf.keras.preprocessing.image.img_to_array(img_flip)
    
    # 3. Brightened
    img_bright = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10)
    img_array3 = tf.keras.preprocessing.image.img_to_array(img_bright)
    
    # Stack into a batch of 3
    batch = np.stack([img_array1, img_array2, img_array3])
    
    # Predict all at once natively (faster)
    predictions = model.predict(batch, verbose=0)
    
    # Average the confidence scores to make it much more stable
    avg_pred = np.mean(predictions, axis=0)
    
    # Get the highest probability class
    class_idx = np.argmax(avg_pred)
    confidence = np.max(avg_pred) * 100
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
        # Clean up the weird class names (e.g., Tomato__Target_Spot -> Target Spot)
        disease = class_name.replace("Tomato_", "").replace("__", "_").replace("_", " ").strip()
        disease = disease.replace("  ", " ") # Remove double spaces for safety
        return f"Diseased: {disease}", (0, 0, 255) # Red text

def is_leaf_present(frame):
    """
    A lightning-fast 'pre-check' to see if there's actually a plant/leaf in the image.
    It checks if there's a minimum amount of green/yellow-green color in the frame.
    """
    # Convert image to HSV color space (easier to filter colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define a much broader color range for leaves to prevent normal plants from getting blocked.
    # Hue: 25 to 90, Saturation: >20, Value: >20 
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask of the green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate what percentage of the image is leaf-colored
    total_pixels = frame.shape[0] * frame.shape[1]
    green_ratio = cv2.countNonZero(mask) / total_pixels
    
    # If more than 1% of the image is green, we say there's a leaf.
    return green_ratio > 0.01

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
        
        # If confidence is low, it might be a normal houseplant or unrelated object
        if confidence < 65.0:
            label_text = "Uncertain / Unrelated Plant"
            color = (0, 165, 255) # Orange Warning
            
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
            
            # If confidence is low, it might be an unrelated plant
            if confidence < 65.0:
                label_text = "Uncertain / Unrelated Plant"
                color = (0, 165, 255)
            
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

def detect_from_directory(model):
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"\nCreated directory: {img_dir}")
        print("Please add some images to this directory and try again.")
        return
        
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(img_dir) if any(f.lower().endswith(ext) for ext in valid_exts)]
    
    if not image_files:
        print(f"\nNo images found in {img_dir}. Please add some images.")
        return
        
    print(f"\nFound {len(image_files)} images in {img_dir}.")
    cv2.namedWindow("Batch Detection", cv2.WINDOW_NORMAL)
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        print(f"\n--- Processing: {img_file} ---")
        
        frame = cv2.imread(img_path)
        if frame is None:
            print("Failed to load image.")
            continue
            
        # PRE-CHECK: Is there a leaf in the image?
        if not is_leaf_present(frame):
            label_text = "No Leaf Detected!"
            color = (0, 165, 255) # Orange for warning
            confidence = 0.0
            class_name = "N/A"
            print("Status: No Plant/Leaf found in image.")
        else:
            class_name, confidence = predict_frame(model, frame)
            label_text, color = get_disease_info(class_name)
            
            if confidence < 65.0:
                label_text = "Uncertain / Unrelated Plant"
                color = (0, 165, 255)
            
            print(f"Status: {label_text}")
            print(f"Confidence: {confidence:.2f}%")
            
        cv2.putText(frame, f"{label_text} ({confidence:.1f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    
        cv2.imshow("Batch Detection", frame)
        print("Press any key to show next image, or 'q' to stop batch detection.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Batch detection stopped.")
            break
            
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
        print("3. Detect from 'assets' directory")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            detect_from_image(model)
        elif choice == '2':
            detect_live(model)
        elif choice == '3':
            detect_from_directory(model)
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
