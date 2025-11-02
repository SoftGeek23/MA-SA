# Setup Guide for MemAgent

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

## Configuration

Edit `configs/default_config.yaml` to customize:
- Agent settings (sleep interval, timeouts)
- Environment settings (browser, viewport)
- Memory settings (embedding model, FAISS index)
- World model settings (architecture, training)

## Usage

### Running the Agent

```bash
python main.py --mode run
```

### Training the World Model

After collecting episodes, train the world model:
```bash
python main.py --mode train_world_model --epochs 10
```

### Using with Your Own LLM

Edit `main.py` or `example_usage.py` to replace `dummy_llm_callback` with your actual 14B model inference.

Example using transformers:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model")
model = AutoModelForCausalLM.from_pretrained("your-model")

def llm_callback(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Architecture Overview

### Components

1. **WebEnvironment**: Playwright-based web automation
2. **EpisodeBuffer**: Collects and manages (state, action, next_state, outcome) tuples
3. **ReflectionSystem**: Extracts rules/anti-rules every 150 episodes
4. **EpisodicMemory**: FAISS-based vector search over state+reflection embeddings
5. **WorldModel**: Predicts next states from state+action pairs
6. **ActionSelector**: Uses retrieved memories to guide action selection

### Data Flow

1. Agent interacts with web environment
2. Episodes are recorded: (state, action, next_state, outcome)
3. Every 150 episodes: sleep/reflection phase
   - Extract rules from successful episodes
   - Extract anti-rules from failed episodes
   - Generate reflections for each state
   - Populate FAISS index with state+reflection embeddings
4. Action selection: kNN retrieval + world model predictions (optional)

## Next Steps

1. Replace dummy LLM callback with your 14B model
2. Create task definitions for your specific web tasks
3. Run agent to collect episodes
4. Train world model on collected episodes
5. Enable world model in action selector for better predictions

