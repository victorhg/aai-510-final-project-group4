import pytest
import torch
import numpy as np
import platform
import os
import sys

# Add the parent directory to the path to allow imports from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import TensorFlow, but handle gracefully if not installed
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.utils.config import config


def test_config_singleton_exists():
    """Test that the config singleton is created and accessible."""
    assert config is not None
    assert hasattr(config, "device")
    assert hasattr(config, "device_name")
    assert hasattr(config, "tf_device_name")
    assert hasattr(config, "keras_device")
    assert hasattr(config, "configure_keras")


def test_pytorch_device_detection():
    """Test that PyTorch can use the detected device."""
    # Check device properties
    assert config.device is not None
    assert isinstance(config.device, torch.device)
    assert config.device_name is not None
    
    # Create a tensor and move it to the device
    tensor = torch.ones((2, 3))
    tensor = tensor.to(config.device)
    
    # Verify the tensor is on the correct device type (not exact device due to index differences)
    assert tensor.device.type == config.device.type
    
    # Perform a simple operation to verify device works
    result = tensor * 2
    assert torch.all(torch.eq(result, torch.full((2, 3), 2, device=config.device)))
    
    # Print device info for debugging
    print(f"PyTorch using: {config.device_name}")


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_tensorflow_device_usage():
    """Test that TensorFlow can use the configured device."""
    # Check TF device properties
    assert config.tf_device_name is not None
    
    # Verify TensorFlow sees the appropriate devices
    tf_devices = tf.config.list_physical_devices()
    print(f"TensorFlow devices: {tf_devices}")
    
    # Run a simple TF operation on the configured device
    with tf.device(config.tf_device_name):
        x = tf.ones((2, 3))
        y = x * 2
        
        # Convert to numpy to verify values
        result = y.numpy()
        assert np.array_equal(result, np.full((2, 3), 2))
    
    # Print device info for debugging
    print(f"TensorFlow using: {config.tf_device_name}")


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_keras_configuration():
    """Test Keras configuration works correctly."""
    # Configure Keras with our settings
    result = config.configure_keras()
    assert result is True
    
    # Create a simple model using the functional API to avoid warnings
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(4, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy data
    x = np.random.random((10, 3))
    y = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(10,)), num_classes=2)
    
    # Verify we can fit for one batch (without throwing device errors)
    model.fit(x, y, epochs=1, batch_size=10, verbose=0)
    
    # Verify we can make predictions
    predictions = model.predict(x, verbose=0)
    assert predictions.shape == (10, 2)
    
    print(f"Keras configured successfully on: {config.keras_device}")


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_tensorflow_and_pytorch_compatibility():
    """Test that both frameworks can be used together with the same config."""
    # PyTorch tensor
    pt_tensor = torch.ones((2, 3), device=config.device)
    pt_result = pt_tensor * 3
    
    # TensorFlow tensor
    with tf.device(config.tf_device_name):
        tf_tensor = tf.ones((2, 3))
        tf_result = tf_tensor * 3
    
    # Verify both frameworks produced the same result
    np_pt = pt_result.detach().cpu().numpy()
    np_tf = tf_result.numpy()
    
    assert np.array_equal(np_pt, np_tf)
    assert np.array_equal(np_pt, np.full((2, 3), 3))


def test_gpu_detection_consistency():
    """Test that the has_gpu flag is consistent with the device type."""
    if config.has_gpu:
        assert config.device.type in ["cuda", "mps", "hip", "xpu"]
    else:
        assert config.device.type == "cpu" 