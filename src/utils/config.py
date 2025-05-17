import torch
import platform
import os

class Config:
    """
    Configuration class that automatically detects and selects the best available
    compute device for deep learning operations.
    
    This class creates a singleton configuration instance that determines whether
    to use CUDA (NVIDIA), MPS (Apple Silicon), ROCm (AMD), specialized accelerators,
    or CPU as a fallback.
    
    Supports PyTorch, TensorFlow, and Keras device configurations.
    
    Example usage:
        from src.utils.config import config  # Add project root to PYTHONPATH first
        
        # PyTorch usage
        print(f"Using PyTorch device: {config.device_name}")
        x = torch.randn(3, 3)
        x = x.to(config.device)
        
        # TensorFlow usage
        import tensorflow as tf
        with tf.device(config.tf_device_name):
            y = tf.random.normal((3, 3))
            
        # Keras usage
        model = tf.keras.Sequential([...])
        config.configure_keras()  # Set up Keras to use the detected device
    """
    
    def __init__(self):
        self._device = self._get_device()
        # Set TensorFlow environment variables based on detected hardware
        self._configure_tensorflow_env()

    def _get_device(self):
        """
        Determine the best available device in order of prevalence and performance:
        1. CUDA (NVIDIA GPUs) - Most common in ML development environments
        2. MPS (Apple Silicon GPUs) - Increasingly common with M1/M2/M3 Macs
        3. ROCm (AMD GPUs) - Less common but growing in ML workstations
        4. XPU (Intel/Graphcore IPUs) - Specialized hardware, relatively rare
        5. CPU (fallback) - Available on all systems
        
        This order prioritizes both prevalence in ML workstations and performance.
        """
        # NVIDIA GPUs - Most common and best supported for ML
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon GPUs - Growing user base among ML developers
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        # AMD GPUs - Less common but growing support
        elif hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available') and torch.hip.is_available():
            return torch.device("hip")
        # Specialized hardware - Relatively rare
        elif hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available') and torch.xpu.is_available():
            return torch.device("xpu")
        # CPU - Always available as fallback
        else:
            return torch.device("cpu")
            
    def _configure_tensorflow_env(self):
        """Configure TensorFlow environment variables based on available hardware"""
        if self._device.type == "cuda":
            # For NVIDIA GPUs
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        elif self._device.type == "mps":
            # For Apple Silicon
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            if not os.environ.get('CUDA_VISIBLE_DEVICES'):
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA for TF when using MPS
        elif self._device.type == "hip":
            # For AMD GPUs
            os.environ['TF_ROCM_FUSION_ENABLE'] = '1'
        else:
            # For CPU only - disable GPU
            if not os.environ.get('CUDA_VISIBLE_DEVICES'):
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def configure_keras(self):
        """
        Configure Keras to use the detected device.
        This should be called before creating and training Keras models.
        
        For TensorFlow 2+, Keras is integrated with TensorFlow, so this mainly
        sets the appropriate memory growth settings and visible devices.
        """
        try:
            import tensorflow as tf
            
            # Configure memory growth to avoid allocating all GPU memory at once
            if self._device.type == "cuda" or self._device.type == "hip":
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        
            # Set visible devices appropriately
            if self._device.type == "cpu" or self._device.type == "mps":
                # Hide GPUs from TensorFlow on CPU or MPS (Apple Silicon)
                tf.config.set_visible_devices([], 'GPU')
                
            # For mixed precision training on NVIDIA GPUs
            if self._device.type == "cuda" and hasattr(tf, 'keras') and hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                if tf.config.list_physical_devices('GPU'):
                    # Use mixed precision on NVIDIA GPUs with compute capability 7.0+
                    # Improves performance on Volta, Turing, and Ampere architectures
                    policy = 'mixed_float16'
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print(f"Keras mixed precision policy set to: {policy}")
                    
            return True
        except ImportError:
            print("TensorFlow/Keras not found. Install with: pip install tensorflow")
            return False
        except Exception as e:
            print(f"Error configuring Keras: {str(e)}")
            return False

    @property
    def device(self):
        """Get the device to use for PyTorch operations"""
        return self._device
        
    @property
    def device_name(self):
        """Get a human-readable description of the device for PyTorch"""
        if self._device.type == "cuda":
            return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        elif self._device.type == "mps":
            return f"Apple Silicon GPU"
        elif self._device.type == "hip":
            return f"AMD GPU (ROCm)"
        elif self._device.type == "xpu":
            return f"Graphcore/Intel IPU"
        else:
            return f"CPU ({platform.processor()})"
            
    @property
    def tf_device_name(self):
        """Get the device string to use for TensorFlow/Keras operations"""
        if self._device.type == "cuda":
            return "/GPU:0"
        elif self._device.type == "mps":
            # TensorFlow doesn't directly support MPS, use CPU with optimizations
            return "/CPU:0"
        elif self._device.type == "hip":
            return "/GPU:0"  # TensorFlow with ROCm uses the same device name as CUDA
        elif self._device.type == "xpu":
            # This would depend on TF XPU plugin configuration
            return "/XPU:0" if "XPU" in os.environ.get("TF_DEVICE_ORDER", "") else "/CPU:0"
        else:
            return "/CPU:0"
    
    @property
    def has_gpu(self):
        """Returns True if any GPU is available (CUDA, MPS, or ROCm)"""
        return self._device.type in ["cuda", "mps", "hip", "xpu"]
    
    @property
    def keras_device(self):
        """Get device string for Keras backend"""
        return self.tf_device_name  # Same as TensorFlow in TF 2.x
            
    def __str__(self):
        return f"Config(pytorch_device={self.device_name}, tensorflow_device={self.tf_device_name})"


# Create a singleton instance
config = Config() 