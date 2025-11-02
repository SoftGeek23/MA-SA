# MemAgent - Agent Learning from Runtime Experience

An agent architecture that learns from deterministic web task interactions through episodic memory, reflection-based rule extraction, and implicit world modeling.

## Architecture

The system consists of:
- **Web Environment**: Playwright-based deterministic web task execution
- **Episode Recording**: Captures (state, action, next_state, outcome) tuples
- **Reflection System**: Extracts rules and anti-rules every 150 episodes
- **FAISS Episodic Memory**: Vector search over state+reflection embeddings
- **Implicit World Model**: Predicts next states from state+action pairs

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```

2. Configure settings in `configs/default_config.yaml`

3. Run the agent:
```bash
python main.py
```

## Project Structure

See the plan document for detailed architecture.

