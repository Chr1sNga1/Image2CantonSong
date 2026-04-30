"""
CLIP-E Crossentropy Model Inference Script
Predicts sentiment labels from images using the CLIP-E crossentropy model.
"""

import argparse
import sys
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import AutoProcessor, TFCLIPModel
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from paths import PROJECT_ROOT


class CLIPECrossEntropy:
    """CLIP-E Crossentropy model for sentiment classification."""
    
    # Sentiment label mappings
    LABELS_25 = ['affection', 'cheerfullness', 'confusion', 'contentment', 'disappointment', 
                 'disgust', 'enthrallment', 'envy', 'exasperation', 'gratitude', 'horror', 
                 'irritabilty', 'lust', 'neglect', 'nervousness', 'optimism', 'pride', 'rage',
                 'relief', 'sadness', 'shame', 'suffering', 'surprise', 'sympathy', 'zest']
    
    LABELS_6 = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    
    LABELS_BINARY = ['negative', 'positive']
    
    def __init__(self, model_path_25cat=None, model_path_6cat=None, model_path_binary=None, verbose=True):
        """
        Initialize CLIP-E model with weights.
        
        Args:
            model_path_25cat: Path to 25-category model weights
            model_path_6cat: Path to 6-category model weights
            model_path_binary: Path to binary model weights
            verbose: Whether to print model loading messages
        """
        self.verbose = verbose
        # Load CLIP model and processor
        if self.verbose:
            print("Loading CLIP model and processor...")
        self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize models
        self.model_25cat = None
        self.model_6cat = None
        self.model_binary = None
        
        # Load model weights if provided
        if model_path_25cat and Path(model_path_25cat).exists():
            self.model_25cat = self._build_model(num_classes=25)
            self.model_25cat.load_weights(model_path_25cat)
            if self.verbose:
                print(f"Loaded 25-category model from {model_path_25cat}")
        
        if model_path_6cat and Path(model_path_6cat).exists():
            self.model_6cat = self._build_model(num_classes=6)
            self.model_6cat.load_weights(model_path_6cat)
            if self.verbose:
                print(f"Loaded 6-category model from {model_path_6cat}")
        
        if model_path_binary and Path(model_path_binary).exists():
            self.model_binary = self._build_model(num_classes=1, activation='sigmoid')
            self.model_binary.load_weights(model_path_binary)
            if self.verbose:
                print(f"Loaded binary model from {model_path_binary}")
    
    @staticmethod
    def _build_model(num_classes=25, activation='softmax'):
        """
        Build CLIP-E crossentropy model.
        
        Args:
            num_classes: Number of output classes
            activation: Activation function for output layer
        
        Returns:
            Compiled Keras model
        """
        IMG_FEATURES_SIZE = 512
        INPUT_img = tf.keras.layers.Input(shape=(IMG_FEATURES_SIZE,), name='input_img_features')
        fc = tf.keras.layers.Dense(512, activation='relu', name='img_fc1')(INPUT_img)
        preds = tf.keras.layers.Dense(num_classes, activation=activation, name='preds')(fc)
        
        model = tf.keras.models.Model(inputs=INPUT_img, outputs=preds)
        return model
    
    def _extract_image_features_from_pil(self, image: Image.Image):
        """
        Extract image features from a PIL Image.
        
        Args:
            image: PIL Image
        
        Returns:
            Image embeddings (numpy array)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="tf")
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.numpy()

    def _extract_image_features_from_path(self, image_path: str):
        """
        Extract image features from an image file path.
        """
        with Image.open(image_path) as image:
            return self._extract_image_features_from_pil(image)

    def _extract_image_features_from_bytes(self, image_bytes: bytes):
        """
        Extract image features from raw image bytes.
        """
        with Image.open(BytesIO(image_bytes)) as image:
            return self._extract_image_features_from_pil(image)

    def _extract_image_features(self, image):
        """
        Extract image features using the appropriate input type.
        
        Args:
            image: PIL Image, image file path, or raw image bytes
        
        Returns:
            Image embeddings (numpy array)
        """
        if isinstance(image, Image.Image):
            return self._extract_image_features_from_pil(image)
        if isinstance(image, str):
            return self._extract_image_features_from_path(image)
        if isinstance(image, (bytes, bytearray)):
            return self._extract_image_features_from_bytes(bytes(image))
        raise TypeError("Unsupported image input type")
    
    def predict_top_n_from_pil(self, image: Image.Image, n=5, model_type='25cat'):
        """
        Predict top n sentiment labels from a PIL Image.
        """
        return self.predict_top_n(image, n=n, model_type=model_type)

    def predict_top_n_from_path(self, image_path: str, n=5, model_type='25cat'):
        """
        Predict top n sentiment labels from an image file path.
        """
        with Image.open(image_path) as image:
            return self.predict_top_n_from_pil(image, n=n, model_type=model_type)

    def predict_top_n_from_bytes(self, image_bytes: bytes, n=5, model_type='25cat'):
        """
        Predict top n sentiment labels from raw image bytes.
        """
        with Image.open(BytesIO(image_bytes)) as image:
            return self.predict_top_n_from_pil(image, n=n, model_type=model_type)

    def predict_from_pil(self, image: Image.Image, model_type='25cat'):
        """
        Predict sentiment scores for a PIL Image using the full label set.
        """
        return self.predict(image, model_type=model_type)

    def predict_from_path(self, image_path: str, model_type='25cat'):
        """
        Predict sentiment scores from an image file path using the full label set.
        """
        with Image.open(image_path) as image:
            return self.predict_from_pil(image, model_type=model_type)

    def predict_from_bytes(self, image_bytes: bytes, model_type='25cat'):
        """
        Predict sentiment scores from raw image bytes using the full label set.
        """
        with Image.open(BytesIO(image_bytes)) as image:
            return self.predict_from_pil(image, model_type=model_type)

    def predict_top_n(self, image, n=None, model_type='25cat'):
        """
        Predict the top n sentiment labels for an image.
        
        Args:
            image: PIL Image or path to image file
            n: Number of top predictions to return, or None to return the full list
            model_type: Type of model to use ('25cat', '6cat', or 'binary')
        
        Returns:
            List of dictionaries with keys `label` and `score`, sorted by score descending.
        """
        results = self.predict(image, model_type=model_type)
        if n is None:
            return results
        if n <= 0:
            return []
        return results[:n]
    
    def predict(self, image, model_type='25cat'):
        """
        Predict sentiment scores for the full label set of a single model.
        
        Args:
            image: PIL Image or path to image file
            model_type: Type of model to use ('25cat', '6cat', or 'binary')
        
        Returns:
            List of dictionaries with keys `label` and `score` for every class in the model.
        """
        if model_type == '25cat':
            if self.model_25cat is None:
                raise ValueError("25-category model not loaded")
            model = self.model_25cat
            labels = self.LABELS_25
        elif model_type == '6cat':
            if self.model_6cat is None:
                raise ValueError("6-category model not loaded")
            model = self.model_6cat
            labels = self.LABELS_6
        elif model_type == 'binary':
            if self.model_binary is None:
                raise ValueError("Binary model not loaded")
            model = self.model_binary
            labels = self.LABELS_BINARY
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        image_features = self._extract_image_features(image)
        preds = model.predict(image_features, verbose=0)
        preds = np.squeeze(preds)

        if model_type == 'binary':
            preds = np.array([1 - preds, preds])

        results = [{"label": labels[idx], "score": float(preds[idx])} for idx in range(len(labels))]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-E crossentropy inference for image mood classification."
    )
    parser.add_argument(
        "--stdin-bytes",
        action="store_true",
        help="Read raw image bytes from stdin instead of from a file.",
    )
    parser.add_argument(
        "--model-type",
        choices=["25cat", "6cat", "binary"],
        default="25cat",
        help="Model type to use for prediction.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top labels to print. Omit to print the full sorted label list.",
    )
    parser.add_argument(
        "--mood-only",
        action="store_true",
        help="Only print top labels comma-separated and suppress other outputs.",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the input image file.",
    )

    args = parser.parse_args()

    if args.stdin_bytes:
        image_bytes = sys.stdin.buffer.read()
        if not image_bytes:
            parser.error("No image bytes received on stdin.")
    else:
        if not args.image_path:
            parser.error("image_path is required unless --stdin-bytes is set.")
        with open(args.image_path, "rb") as f:
            image_bytes = f.read()

    model = CLIPECrossEntropy(
        model_path_25cat=PROJECT_ROOT / "Emotion" / "CLIP-E" / 'clip-e_25cat.hdf5',
        model_path_6cat=PROJECT_ROOT / "Emotion" / "CLIP-E" / 'clip-e_6cat.hdf5',
        model_path_binary=PROJECT_ROOT / "Emotion" / "CLIP-E" / 'clip-e_binary.hdf5',
        verbose=not args.mood_only,
    )
    if args.top_n is None:
        results = model.predict_from_bytes(image_bytes, model_type=args.model_type)
    else:
        results = model.predict_top_n_from_bytes(
            image_bytes, n=args.top_n, model_type=args.model_type
        )
    if args.mood_only:
        print(", ".join(item["label"] for item in results))
    else:
        for item in results:
            print(f"{item['label']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()
