# ALFWorld Integration Setup Guide

This guide explains how to set up and use ALFWorld with your agent.

## What is ALFWorld?

ALFWorld is a text-based interactive environment aligned with the ALFRED embodied AI dataset. It allows agents to practice high-level reasoning in abstract text environments before tackling embodied tasks. See [https://alfworld.github.io/](https://alfworld.github.io/) for more details.

## Installation

### 1. Install ALFWorld

```bash
pip install alfworld[full]
```

This installs the full version with all dependencies. For text-only version:
```bash
pip install alfworld
```

### 2. Download ALFWorld Data

ALFWorld requires data files (PDDL files, game files, pre-trained models):

```bash
alfworld-download
```

For additional pre-trained checkpoints:
```bash
alfworld-download --extra
```

By default, data is stored in `~/.cache/alfworld/`. You can customize this by setting:
```bash
export ALFWORLD_DATA=/path/to/your/alfworld_data
```

## Configuration

Edit `configs/default_config.yaml` to enable ALFWorld:

```yaml
alfworld:
  enabled: true  # Set to true to use ALFWorld
  env_type: "AlfredTWEnv"  # Options: AlfredTWEnv, AlfredThorEnv, AlfredHybrid
  data_dir: null  # null for default ~/.cache/alfworld/
  train_eval: "train"  # train or eval split
```

### Environment Types

- **AlfredTWEnv**: TextWorld environment (text-only, faster, recommended for learning)
- **AlfredThorEnv**: AI2-THOR visual environment (requires GPU)
- **AlfredHybrid**: Combined text and visual

For most use cases, start with `AlfredTWEnv`.

## Usage

### Running ALFWorld Tasks

Once ALFWorld is enabled in config:

```bash
python main.py
```

The agent will automatically:
1. Initialize ALFWorld environment
2. Load example ALFWorld tasks
3. Use Llama-3.1-8B (if enabled) to generate text commands
4. Execute commands and learn from experience

### Example ALFWorld Tasks

The default tasks include:
- "examine an alarmclock with the desklamp"
- "pick up a mug and put it in the microwave"

### Custom Tasks

You can create custom ALFWorld tasks in `main.py`:

```python
task_def.create_alfworld_task(
    task_id="custom_task_1",
    description="Your task description",
    goal="your natural language goal here"
)
```

## How It Works

1. **State Representation**: ALFWorld observations are converted to `WebState` format for compatibility with your existing agent architecture

2. **Actions**: The agent generates text commands like:
   - "go to desk 1"
   - "take alarmclock 2 from desk 1"
   - "use desklamp 1"

3. **Action Selection**: Llama-3.1-8B (if enabled) generates commands based on:
   - Current observation/state
   - Task goal
   - Past experience (episodic memory)

4. **Learning**: The agent learns from episodes just like with web tasks:
   - Records (state, action, next_state, outcome)
   - Reflects every 150 episodes
   - Builds episodic memory

## Troubleshooting

### Import Errors

If you see `ImportError: ALFWorld is not installed`:
```bash
pip install alfworld[full]
```

### Data Not Found

If ALFWorld can't find data files:
```bash
alfworld-download
```

Check that data is in `~/.cache/alfworld/` or set `ALFWORLD_DATA` environment variable.

### Out of Memory

ALFWorld text environments are lightweight, but if using `AlfredThorEnv` (visual), ensure you have:
- GPU with at least 12GB memory (GTX 1080 Ti or better)
- Sufficient system RAM

### Task Completion Issues

ALFWorld tasks use natural language goals. The agent may need several attempts to learn effective command sequences. This is expected and part of the learning process.

## Differences from Web Environment

1. **Text-based**: Commands are natural language text, not DOM interactions
2. **Admissible Commands**: ALFWorld provides valid commands for each state
3. **Episode Format**: Actions use `type: "text_command"` instead of `click`, `type`, etc.
4. **State**: Uses observation text instead of DOM tree

The agent architecture remains the same - it learns from experience, builds episodic memory, and reflects on past episodes.

## Next Steps

- Run the agent to collect episodes
- Train the world model on collected episodes
- Experiment with different task types
- Try different ALFWorld environment types

For more information, see:
- [ALFWorld Website](https://alfworld.github.io/)
- [ALFWorld GitHub](https://github.com/alfworld/alfworld)
- [ALFRED Dataset](https://askforalfred.com/)

