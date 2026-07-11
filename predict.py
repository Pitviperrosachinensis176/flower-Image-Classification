import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

MODEL_PATH = "flower_model.h5"
IMG_SIZE = (150, 150)
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def load_class_indices(file_path='class_indices.txt'):

    if not os.path.exists(file_path):
        print(f"[WARN] File {file_path} not found... Using default classes.")
        return CLASS_NAMES

    class_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            idx, cls = line.strip().split(':')
            class_map[int(idx)] = cls

    return [class_map[i] for i in sorted(class_map.keys())]


def preprocess_image(image_path, target_size=IMG_SIZE):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, target_size)

    image_normalized = image_resized.astype('float32') / 255.0

    image_batch = np.expand_dims(image_normalized, axis=0)

    return image_batch, image_rgb


def predict_flower(image_path, model_path=MODEL_PATH, show=True):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run train.py first.")

    class_names = load_class_indices()

    print(f"[INFO] Loading model from {model_path}...")
    model = load_model(model_path)

    print(f"[INFO] Preprocessing image {image_path}...")
    image_batch, image_original = preprocess_image(image_path)

    print("[INFO] Loading model from {model_path}...")
    predictions = model.predict(image_batch, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx])
    predicted_class = class_names[predicted_idx]

    result = {
        'class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }
    }

    print("\n" + "=" * 50)
    print(f"Prediction result: {predicted_class.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)
    print("\nClass probabilities:")
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for cls, prob in sorted_probs:
        bar = '█' * int(prob * 30)
        print(f"  {cls:12s} {prob * 100:6.2f}% {bar}")

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.imshow(image_original)
        ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence * 100:.2f}%',
                      fontsize=14, fontweight='bold')
        ax1.axis('off')

        classes = list(result['all_probabilities'].keys())
        probs = list(result['all_probabilities'].values())
        colors = ['#4CAF50' if c == predicted_class else '#2196F3' for c in classes]

        ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.set_xlim(0, 1)
        for i, v in enumerate(probs):
            ax2.text(v + 0.01, i, f'{v * 100:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\n[INFO] Prediction result saved to prediction_result.png.")

    return result


def main():
    parser = argparse.ArgumentParser(description='Predict flower type using a trained CNN model')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help=f'Path to the model file (default: {MODEL_PATH})')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the prediction result window')

    args = parser.parse_args()

    try:
        predict_flower(args.image, args.model, show=not args.no_show)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
