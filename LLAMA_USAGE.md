# Using Llama 3.1-8B with Base Agent

This guide explains how to use the Llama 3.1-8B model with the base agent.

## Prerequisites

1. **Accept the License**: First, you need to accept the Llama 3.1 license agreement on Hugging Face:
   - Visit: https://huggingface.co/meta-llama/Llama-3.1-8B
   - Click "Agree and access repository"
   - You'll need a Hugging Face account (sign up if needed)

2. **Get Hugging Face Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" access
   - Copy the token

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: `bitsandbytes` is optional but recommended for GPU users (reduces memory usage with 4-bit quantization).

## Usage Methods

### Method 1: Auto-initialization via Config (Recommended)

1. **Set your Hugging Face token** (choose one):
   ```bash
   export HUGGINGFACE_TOKEN="your_token_here"
   ```
   
   OR edit `configs/default_config.yaml`:
   ```yaml
   llm:
     use_auth_token: "your_token_here"
   ```

2. **Enable Llama in config** (`configs/default_config.yaml`):
   ```yaml
   llm:
     enabled: true  # Change from false to true
     model_name: "meta-llama/Llama-3.1-8B"
     use_quantization: true  # Use 4-bit quantization (CUDA only)
     device: null  # Auto-detect (cuda/cpu)
   ```

3. **Create agent** - it will auto-initialize:
   ```python
   from src.utils.config import Config
   from src.agent.base_agent import BaseAgent
   
   config = Config.from_yaml("configs/default_config.yaml")
   agent = BaseAgent(config)  # Llama will be auto-initialized
   ```

### Method 2: Manual Initialization

```python
from src.utils.config import Config
from src.agent.base_agent import BaseAgent
from src.utils.llama_model import create_llama_callback

# Create Llama callback
llm_callback = create_llama_callback(
    model_name="meta-llama/Llama-3.1-8B",
    use_auth_token="your_token_here",  # or use os.getenv("HUGGINGFACE_TOKEN")
    use_quantization=True,  # Reduces memory usage (CUDA only)
    device="cuda"  # or "cpu", or None for auto-detect
)

# Create agent with callback
config = Config.from_yaml("configs/default_config.yaml")
agent = BaseAgent(config, llm_callback=llm_callback)
```

### Method 3: Using the LlamaModel class directly

```python
from src.utils.llama_model import LlamaModel

model = LlamaModel(
    model_name="meta-llama/Llama-3.1-8B",
    use_auth_token="your_token_here",
    use_quantization=True,
    device="cuda"
)

# Use as callback
response = model("Your prompt here")

# Or use generate method
response = model.generate("Your prompt here", max_new_tokens=512, temperature=0.7)
```

## Configuration Options

The LLM configuration in `configs/default_config.yaml` supports:

```yaml
llm:
  model_name: "meta-llama/Llama-3.1-8B"  # Model name or path
  enabled: false  # Enable/disable auto-initialization
  device: null  # "cuda", "cpu", or null for auto-detect
  use_quantization: true  # 4-bit quantization (requires CUDA and bitsandbytes)
  use_auth_token: null  # HF token or use HUGGINGFACE_TOKEN env var
  max_new_tokens: 512  # Maximum tokens to generate
  temperature: 0.7  # Sampling temperature (0.0 = deterministic)
  do_sample: true  # Use sampling vs greedy decoding
```

## Memory Requirements

- **Full precision (no quantization)**: ~16GB GPU memory
- **4-bit quantization**: ~5GB GPU memory (recommended if you have CUDA)
- **CPU**: Works but will be slow (~1-2 minutes per generation)

## Troubleshooting

1. **Authentication Error**: 
   - Make sure you've accepted the license on Hugging Face
   - Verify your token is correct
   - Check token has "Read" access

2. **Out of Memory**:
   - Enable quantization: `use_quantization: true` (CUDA only)
   - Use CPU: `device: "cpu"` (will be slower)
   - Reduce `max_new_tokens`

3. **bitsandbytes not available**:
   - Only needed for quantization on CUDA
   - Will fall back to full precision if not available

4. **Model Download Issues**:
   - First run will download ~16GB model
   - Ensure stable internet connection
   - Model is cached after first download

## Example Usage

See `example_usage.py` for a complete example. To use with Llama:

```python
from src.utils.llama_model import create_llama_callback
from src.utils.config import Config
from src.agent.base_agent import BaseAgent
import os

# Get token from environment
token = os.getenv("HUGGINGFACE_TOKEN")

# Create callback
llm_callback = create_llama_callback(
    use_auth_token=token,
    use_quantization=True
)

# Create agent
config = Config.from_yaml("configs/default_config.yaml")
agent = BaseAgent(config, llm_callback=llm_callback)

# Use agent as normal
# ... rest of your code
```

