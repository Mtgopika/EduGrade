import os
import cv2
import tensorflow as tf
import numpy as np
from pdf2image import convert_from_path
from PIL import ImageOps, Image
from tensorflow import keras
from keras.layers import StringLookup
from keras import ops
from pathlib import Path
from tensorflow.keras.models import load_model



np.random.seed(42)
keras.utils.set_random_seed(42)


# Ensure the PDF path is correct
POPPLER_PATH = os.getenv("POPPLER_PATH", "C:\\Users\\HP\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin")
images_dir='C:\\Users\\HP\\OneDrive\\Desktop\\edugrade\\edugrade\\edugrade_app\\backend\\images'
labels_dir ='C:\\Users\\HP\\OneDrive\\Desktop\\edugrade\\edugrade\\edugrade_app\\backend\\labels.txt'
def get_image_path(image_dir, image_name):
    """
    Check for the image file with .jpg or .png extensions.
    Returns the full path if the image exists, otherwise None.
    """
    for ext in ['.jpg', '.png']:
        path = Path(image_dir) / f"{image_name}{ext}"
        if path.exists():
            return str(path)
    return None

def load_transcriptions(labels_dir, image_dir):
   
    transcriptions = {}
    current_image_name = None
    current_transcription = ""

    with open(labels_dir, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                continue

            if "," in line:  # New image name and transcription
                # Save the previous transcription if a new image is encountered
                if current_image_name:
                    transcriptions[current_image_name] = current_transcription.strip()

                # Split into image name and text
                try:
                    base_image_name, current_transcription = line.split(",", 1)
                    base_image_name = base_image_name.strip()
                    current_transcription = current_transcription.strip()

                    # Resolve full image path with correct extension
                    current_image_name = get_image_path(image_dir, base_image_name)
                    if not current_image_name:
                        print(f"Image file not found for: {base_image_name}")
                        continue
                except ValueError:
                    print(f"Skipping malformed line: {line}")
                    continue
            else:  # Continuation of the previous transcription
                current_transcription += " " + line.strip()

        # Save the last transcription in the file
        if current_image_name:
            transcriptions[current_image_name] = current_transcription.strip()

    return transcriptions

# Path to your custom dataset
base_path = "C:\\Users\\HP\\OneDrive\\Desktop\\edugrade\\edugrade\\edugrade_app\\backend"
labels_path = os.path.join(base_path, "labels.txt")

words_list = []
current_image_name = None
current_transcription = ""

with open(labels_path, "r") as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:  # Skip empty lines
            continue

        if "," in line:  # New image-transcription pair
            # Save the current transcription if there's an ongoing pair
            if current_image_name:
                words_list.append((current_image_name, current_transcription.strip()))

            # Extract new image name and transcription
            try:
                current_image_name, current_transcription = line.split(",", 1)
                current_image_name = current_image_name.strip()
                current_transcription = current_transcription.strip()
            except ValueError:
                print(f"Skipping malformed line: {line}")
                current_image_name = None
                current_transcription = ""
        else:  # Continuation line
            if current_transcription:
                current_transcription += " " + line.strip()

    # Add the last transcription after finishing the file
    if current_image_name:
        words_list.append((current_image_name, current_transcription.strip()))

# Split the dataset into training, validation, and test sets
split_idx = int(0.9 * len(words_list))  # 90% for training
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]  # Remaining 10% for validation and testing

val_split_idx = int(0.5 * len(test_samples))  # 50% of the 10% for validation
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]  # Remaining 50% for testing

# Ensure the splits are consistent
assert len(words_list) == len(train_samples) + len(validation_samples) + len(test_samples)

base_image_path = os.path.join(base_path, "images")  # Path to your images folder

def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []

    # Possible extensions for images
    valid_extensions = ['.jpg', '.png']

    for image_name, transcription in samples:
        img_path = None

        # Check for the file with valid extensions
        for ext in valid_extensions:
            potential_path = os.path.join(base_image_path, f"{image_name}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path:
            paths.append(img_path)
            corrected_samples.append(transcription)  # Store the transcription
        else:
            print(f"Image not found for: {image_name}")  # Log missing images

    return paths, corrected_samples

# Prepare data splits using the function
train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

characters = sorted(list(characters))

def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        # Strip leading and trailing whitespace
        label = label.strip()
        cleaned_labels.append(label)
    return cleaned_labels

# Clean validation and test labels
validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)

AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

import tensorflow as tf

# Constants
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32
max_len = 32  # Define this based on your maximum label length
AUTOTUNE = tf.data.AUTOTUNE



def preprocess_image(image_path, img_size=(image_width, image_height)):
   
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32)

    
    threshold = 128.0
    image = tf.where(image > threshold, 255.0, 0.0)

   
    image = distortion_free_resize(image, img_size)

    
    image = image / 255.0

    return image


# Vectorize label
def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label

def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}

# Prepare dataset
def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)



class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
    
validation_images = []
validation_labels = []

for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])
    
def calculate_edit_distance(labels, predictions):
    
    saprse_labels = ops.cast(tf.sparse.from_dense(labels), dtype=tf.int64)
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.ops.nn.ctc_decode(
        predictions, sequence_lengths=input_len
    )[0][0][:, :max_len]
    sparse_predictions = ops.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )
# Load OCR model with custom layer
model = load_model('handwritingg.h5', custom_objects={'CTCLayer': CTCLayer})

# Extract the prediction model
prediction_model = keras.models.Model(
    model.get_layer(name="image").output, model.get_layer(name="dense2").output
)

def remove_horizontal_lines(image_np, line_thickness=2):
    """Remove horizontal lines from an image using morphological operations."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Adjust kernel width if needed
    detect_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Subtract lines from the original image
    cleaned = cv2.bitwise_not(binary)
    cleaned = cv2.bitwise_or(cleaned, detect_lines)
    cleaned = cv2.bitwise_not(cleaned)

    # Convert back to 3-channel RGB
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return cleaned_rgb


def correct_skew(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours and compute rotation angle
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        angle = cv2.minAreaRect(largest_contour)[-1]
        
        if angle < -45:
            angle += 90  # Adjust for extreme angles

        # Apply rotation
        (h, w) = binary_image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(binary_image, rotation_matrix, (w, h),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated_image)
    
    return image  # Return unchanged if no contours found

# Convert PDF to images
def process_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"📂 Processing PDF: {pdf_path}")

    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        if not images:
            raise RuntimeError("Poppler failed to extract images")
        return images
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {str(e)}")

# Preprocess image for OCR
def preprocess_image(image):
    grayscale_image = ImageOps.grayscale(image)
    image_array = np.array(grayscale_image)
    thresholded_image = (image_array > 128) * 255  # Binary thresholding
    return Image.fromarray(thresholded_image.astype(np.uint8))

# Distortion-free resize function
def distortion_free_resize(image, img_size):
    # Get current image dimensions
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    # Ensure dimensions are valid
    if height == 0 or width == 0:
        raise ValueError("Invalid image dimensions: height or width is zero.")

    aspect_ratio = width / height
    new_height, new_width = img_size

    # Preserve aspect ratio
    if aspect_ratio > 1:
        h, w = int(new_height / aspect_ratio), new_width
    else:
        h, w = new_height, int(new_width * aspect_ratio)

    # Ensure dimensions are positive
    h = max(1, h)
    w = max(1, w)

    return tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)


# Preprocess the image in memory
def preprocess_image_in_memory(image, img_size=(128, 32)):
    if len(image.shape) == 2:  # Grayscale image
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Apply thresholding to match training
    threshold = 128.0
    image = tf.where(image > threshold, 255.0, 0.0)  

    # Normalize
    image = image / 255.0  

    # Resize with aspect ratio maintained
    image = distortion_free_resize(image, img_size)
    return image


def preprocess_image_for_model(image):
    image = preprocess_image_in_memory(image)  # Call your function
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.transpose(image, perm=[0, 2, 1, 3])  # Ensure correct order
    return image

# Decode predictions properly
def decode_batch_predictions(pred, beam_width=5):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    results = keras.ops.nn.ctc_decode(
        pred, sequence_lengths=input_len, beam_width=beam_width
    )[0][0][:, :max_len]
    
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))  # Remove padding
        text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")

        # Remove unwanted tokens (e.g., [UNK])
        text = text.replace("[UNK]", "").strip()
        
        output_text.append(text)
    return output_text



def extract_characters(image, min_char_width=1, min_char_height=5, min_black_pixels=5):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply soft thresholding (Adaptive instead of Fixed Threshold)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 5)  # Fine-tuned values

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    characters = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # **Adjust the min width & height values**
        if w > min_char_width and h > min_char_height:  # Avoid losing small letters
            char_img = image[y:y+h, x:x+w]  # Extract character region

            # **Ensure it's a valid character (not noise)**
            black_pixels = np.sum(char_img < 128)  # Count non-white pixels
            if black_pixels > min_black_pixels:  # Ignore empty/false contours
                # Resize to fit model input size (128x32)
                char_img = cv2.resize(char_img, (128, 32), interpolation=cv2.INTER_AREA)
                characters.append(char_img)
                bounding_boxes.append((x, y, w, h))

    # **Sort bounding boxes by reading order (left-to-right, top-to-bottom)**
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))
        # **Sort characters and bounding boxes**
    characters = [x for _, x in sorted(zip(bounding_boxes, characters), key=lambda pair: (pair[0][1], pair[0][0]))]

    return characters


# Function to predict text from images
def predict_text_from_images(extracted_words):
    predicted_texts = []
    for word in extracted_words:
        # Preprocess the word image
        word_image = preprocess_image_for_model(word)  

        # Ensure correct dimensions: (batch, height=128, width=32, channels=1)
        word_image = tf.image.resize(word_image, (128, 32))  

        # Check if channel dimension exists; if not, add it
        if len(word_image.shape) == 2:  # If shape = (height, width)
            word_image = tf.expand_dims(word_image, axis=-1)  # Add channel dim

        # Ensure batch dimension is present
        if len(word_image.shape) == 3:  # If shape = (height, width, channels)
            word_image = tf.expand_dims(word_image, axis=0)  # Add batch dim

        preds = prediction_model.predict(word_image)  # Predict
        pred_texts = decode_batch_predictions(preds)  # Decode
        predicted_texts.extend(pred_texts)

    return predicted_texts





