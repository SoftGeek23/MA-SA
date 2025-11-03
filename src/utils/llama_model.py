"""Llama 3.1-8B model wrapper for LLM inference."""
import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional import for quantization
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger(__name__)


class LlamaModel:
    """Wrapper for Llama 3.1-8B model from Hugging Face."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        device: Optional[str] = None,
        use_quantization: bool = True,
        use_auth_token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """Initialize Llama model.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to load model on (None for auto-detect)
            use_quantization: Whether to use 4-bit quantization to reduce memory
            use_auth_token: Hugging Face auth token (required for Llama models)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Llama model {model_name} on {self.device}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=use_auth_token,
                trust_remote_code=True
            )
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Configure model loading
        model_kwargs = {
            "token": use_auth_token,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        # Add quantization config if requested
        if use_quantization and self.device == "cuda":
            if BITSANDBYTES_AVAILABLE:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 4-bit quantization")
                except Exception as e:
                    logger.warning(f"Failed to set up quantization: {e}, skipping")
                    use_quantization = False
            else:
                logger.warning("bitsandbytes not available, skipping quantization")
                use_quantization = False
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if not use_quantization or self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Adjust based on model context window
        ).to(self.device)
        
        # Get generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = kwargs.get("do_sample", self.do_sample)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def __call__(self, prompt: str) -> str:
        """Make the model callable for use as llm_callback.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        return self.generate(prompt)


def create_llama_callback(
    model_name: str = "meta-llama/Llama-3.1-8B",
    device: Optional[str] = None,
    use_quantization: bool = True,
    use_auth_token: Optional[str] = None,
    **kwargs
) -> callable:
    """Create a callback function for Llama model.
    
    This is a convenience function that creates a LlamaModel instance
    and returns it as a callable callback function.
    
    Args:
        model_name: Hugging Face model name
        device: Device to load model on
        use_quantization: Whether to use 4-bit quantization
        use_auth_token: Hugging Face auth token
        **kwargs: Additional parameters for LlamaModel
        
    Returns:
        Callable function that takes a prompt and returns generated text
    """
    model = LlamaModel(
        model_name=model_name,
        device=device,
        use_quantization=use_quantization,
        use_auth_token=use_auth_token,
        **kwargs
    )
    
    return model

