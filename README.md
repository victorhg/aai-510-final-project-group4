# aai510-final-project-group4
USD AAI510 Final Project

Shiley-Marcos School of Engineering, University of San Diego AAI-510: Machine Learning Fundamentals and Application

## Group members

- Christopher Mendoza
- Laurentius Von Liechti
- Victor Hugo Germano 

## Files

[Google Drive Folder with documents and colaboration files](https://drive.google.com/drive/u/2/folders/11tU3hk8GN4HaXU-kp1jkKWC9WO2rZP7o)


## Objective

This project aims to develop a machine learning model that recommends the most suitable crop to cultivate based on specific agro-climatic conditions. Utilizing the Crop Recommendation Dataset from Kaggle, which includes features such as soil nutrients (N, P, K), temperature, humidity, pH, and rainfall, the objectives are:

- Analyze and preprocess the dataset to ensure data quality and suitability for modeling.
- Explore various classification algorithms (e.g., Random Forest, Support Vector Machines, Neural Networks) to identify the most effective model for crop prediction.
- Evaluate model performance using appropriate metrics and validate its predictive accuracy.
- Develop an interpretable and user-friendly recommendation system that can assist farmers in making informed decisions to enhance agricultural productivity.


## References

---

## Technical Details

### Project Structure

```
.
├── src/               # Source code
│   ├── data/          # Data loading and processing
│   ├── models/        # Model definitions
│   └── utils/         # Utility functions
│       └── config.py  # Device configuration
├── tests/             # Test suite
├── .gitignore         # Git ignore rules
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

### Installation

1. Clone the repository
2. Change the current directory to the cloned repo
3. Install dependencies

```bash
pip install -r requirements.txt
```

### Configuration Usage

To use the configuration in scripts, make sure the project root is in your Python path:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add project root to path
from src.utils.config import config

# Check which device was detected
print(f"Using device: {config.device_name}")
print(f"GPU available: {config.has_gpu}")
```

#### PyTorch Example

```python
from src.utils.config import config
import torch

# Create a tensor and move it to the detected device
x = torch.randn(3, 3)
x = x.to(config.device)

# Create a model and move it to the device
model = torch.nn.Sequential(
    torch.nn.Linear(3, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1)
).to(config.device)

# Train with device-aware tensors
inputs = torch.randn(100, 3).to(config.device)
targets = torch.randn(100, 1).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
```

#### TensorFlow Example

```python
from src.utils.config import config
import tensorflow as tf

# Create tensors using TensorFlow on the configured device
with tf.device(config.tf_device_name):
    # Create a simple model
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Generate some data
    input_data = tf.random.normal((100, 3))
    target_data = tf.random.normal((100, 1))
    
    # Train the model (will use the configured device)
    model.fit(input_data, target_data, epochs=5, verbose=0)
    
    # Make predictions
    predictions = model.predict(input_data)
```

#### Keras Example

```python
from src.utils.config import config
import tensorflow as tf

# Configure Keras to use the detected device
config.configure_keras()

# Create a Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Example with MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 784)) / 255.0
test_images = test_images.reshape((10000, 784)) / 255.0

# Training will automatically use the configured device
history = model.fit(
    train_images, train_labels, 
    epochs=5, 
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

Key features of the device configuration:
- **PyTorch**: Automatic device selection (CUDA, MPS, ROCm, CPU)
- **TensorFlow**: Proper device naming and environment variables
- **Keras**: Memory growth settings, mixed precision, and device optimization

### Testing

Run the test suite with pytest:

```bash
pytest
```

Or run tests directly:

```bash
python -m tests.test_config
```
