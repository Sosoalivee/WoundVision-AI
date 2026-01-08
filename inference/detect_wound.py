from ultralytics import YOLO
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_wound.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    # Load the trained model
    model = YOLO("runs/wound_cls4/weights/best.pt")

    # Perform prediction
    results = model.predict(image_path, save=False, verbose=False)

    # Print results
    for result in results:
        probs = result.probs
        if probs is not None:
            # Get the predicted class and confidence
            predicted_class_idx = probs.top1
            confidence = probs.top1conf.item()
            class_names = model.names
            predicted_class = class_names[predicted_class_idx]

            print(f"Predicted Wound Type: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")

            # Print all class probabilities
            print("\nAll Probabilities:")
            for i, prob in enumerate(probs.data):
                print(f"{class_names[i]}: {prob.item():.4f}")
        else:
            print("No prediction results found.")

if __name__ == "__main__":
    main()
