from ultralytics import YOLO
import torch

def train_model():
    # Create a new YOLO model
    model = YOLO('yolov8n.yaml')
    
    # Train the model
    results = model.train(
        data='dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=32,
        name='forest_fire_model_30epochs_batch32',
        patience=20,  # Early stopping patience
        save=True,  # Save best model
        device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    # Validate the model
    results = model.val()
    
    print("Training completed! Model saved in 'runs/detect/forest_fire_model_30epochs_batch32'")

if __name__ == "__main__":
    train_model() 