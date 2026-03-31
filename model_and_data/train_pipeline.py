import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. CONFIGURATION (ADVANCED PIPELINE)
# ==========================================
DATASET_PATH = r"d:\dataset cv\archive (1)\PlantVillage"
MODEL_SAVE_PATH = "tomato_disease_model_efficientnetb3.h5"

# Revert to 224x224 for MobileNetV2 speed
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32 # Can fit 32 again
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

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

def load_and_split_dataset(data_dir, img_size, batch_size, classes):
    print("Loading datasets at High Resolution (300x300)...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        class_names=classes
    )
    
    val_test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        class_names=classes
    )
    
    val_batches = tf.data.experimental.cardinality(val_test_dataset)
    val_dataset = val_test_dataset.take(val_batches // 2)
    test_dataset = val_test_dataset.skip(val_batches // 2)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def build_advanced_model(num_classes, img_size):
    inputs = tf.keras.Input(shape=img_size + (3,))
    
    # Advanced Data Augmentation
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.3)(x)
    x = tf.keras.layers.RandomZoom(0.3)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)
    
    # MobileNetV2 Requires Manual Rescaling! (unlike EfficientNet)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
    
    # MobileNetV2 Base Model 
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Initially freeze all layers of the base model
    base_model.trainable = False 
    
    x = base_model(x, training=False) # Keep BatchNorm layers in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x) # Increased dropout for the larger model
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model, base_model

def get_callbacks():
    """Callbacks for Early Stopping, Learning Rate Reduction, and Model Checkpointing"""
    callbacks = [
        # Reduce LR when validation loss stops dropping
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6, 
            verbose=1
        ),
        # Stop training if validation loss hasn't improved in 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        # ALWAYS save the best version of the model dynamically during training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def plot_combined_history(history_initial, history_fine, initial_epochs):
    acc = history_initial.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_initial.history['loss'] + history_fine.history['loss']
    val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('advanced_training_curves.png')
    plt.show()

def evaluate_model(model, test_dataset, class_names):
    print("\n--- Evaluating Advanced Model on Test Dataset ---")
    loss, accuracy = model.evaluate(test_dataset)
    print(f"\nFinal TEST Accuracy: {accuracy*100:.3f}%\n")
    
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=-1))
        
    display_classes = [c.replace("Tomato_", "").replace("_", " ") for c in class_names]
    print(classification_report(y_true, y_pred, target_names=display_classes, digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_classes)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    plt.title("Advanced Confusion Matrix")
    plt.tight_layout()
    plt.savefig('advanced_confusion_matrix.png')
    plt.show()

def main():
    train_ds, val_ds, test_ds = load_and_split_dataset(
        DATASET_PATH, IMG_SIZE, BATCH_SIZE, TOMATO_CLASSES
    )
    
    model, base_model = build_advanced_model(len(TOMATO_CLASSES), IMG_SIZE)
    callbacks = get_callbacks()
    
    # -------------------------------------------------------------
    # PHASE 1: Train just the new Classification Head (Base frozen)
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Training Top Classification Layers ---")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )
    
    # -------------------------------------------------------------
    # PHASE 2: Fine-Tuning the Deep Base Model
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Unfreezing Base Model for Fine-Tuning ---")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Only keep the top ~50 layers trainable, freeze the deep mathematical layers
    # MobileNetV2 has 154 layers total. We unfreeze from layer 100 onwards.
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a MUCH LOWER learning rate (1e-5 instead of 1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Continue training from the end of phase 1
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history_initial.epoch[-1] + 1,
        callbacks=callbacks
    )
    
    plot_combined_history(history_initial, history_fine, INITIAL_EPOCHS)
    
    # Evaluate final state
    evaluate_model(model, test_ds, TOMATO_CLASSES)

if __name__ == "__main__":
    main()
