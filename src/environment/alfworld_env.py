"""ALFWorld environment wrapper for text-based interactive tasks."""
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    from alfworld.agents.environment import get_environment
    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False

from .state_encoder import WebState, StateEncoder
from .task_definitions import Task


logger = logging.getLogger(__name__)


class ALFWorldEnvironment:
    """ALFWorld environment wrapper that implements the same interface as WebEnvironment."""
    
    def __init__(
        self,
        env_type: str = "AlfredTWEnv",
        data_dir: Optional[str] = None,
        train_eval: str = "train"
    ):
        """Initialize ALFWorld environment.
        
        Args:
            env_type: Type of ALFWorld environment ('AlfredTWEnv', 'AlfredThorEnv', 'AlfredHybrid')
            data_dir: Directory containing ALFWorld data (defaults to ~/.cache/alfworld/)
            train_eval: Whether to use 'train' or 'eval' split
        """
        if not ALFWORLD_AVAILABLE:
            raise ImportError(
                "ALFWorld is not installed. Install it with: pip install alfworld[full]"
            )
        
        self.env_type = env_type
        self.data_dir = data_dir
        self.train_eval = train_eval
        self.env = None
        self.config = None
        self.state_encoder = StateEncoder()
        self.current_task: Optional[Task] = None
        self.current_observation: Optional[str] = None
        self.current_info: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        
        # ALFWorld state tracking
        self.batch_size = 1
        self.done = False
        self.current_score = 0.0
    
    def start(self):
        """Start/initialize the ALFWorld environment."""
        if self.is_initialized:
            logger.debug("ALFWorld already initialized, skipping")
            return
        
        logger.info("Starting ALFWorld environment initialization...")
        
        try:
            # Determine data path - use provided data_dir or ALFWorld default
            import os
            import sys
            from pathlib import Path
            
            if self.data_dir:
                data_path = os.path.expanduser(self.data_dir)
            else:
                # ALFWorld default data directory
                data_path = os.path.expanduser('~/.cache/alfworld')
            
            # Use ALFWorld's config loading mechanism with a config file
            # This is the proper way as per ALFWorld documentation
            config_file = Path(__file__).parent.parent.parent / 'configs' / 'alfworld_config.yaml'
            
            if not config_file.exists():
                logger.warning(f"ALFWorld config file not found: {config_file}")
                logger.info("Creating default ALFWorld config...")
                # Fall back to manual config if file doesn't exist
                self.config = self._create_manual_config(data_path)
            else:
                # Load config YAML directly (simpler than using generic.load_config())
                import yaml
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
                
                # Expand ~ in paths
                if 'env' in self.config and 'data_dir' in self.config['env']:
                    if self.config['env']['data_dir']:
                        self.config['env']['data_dir'] = os.path.expanduser(self.config['env']['data_dir'])
                if 'dataset' in self.config and 'data_path' in self.config['dataset']:
                    if self.config['dataset']['data_path']:
                        self.config['dataset']['data_path'] = os.path.expanduser(self.config['dataset']['data_path'])
                
                # Override data path if specified
                if self.data_dir:
                    self.config['env']['data_dir'] = data_path
                    self.config['dataset']['data_path'] = data_path
                elif 'data_dir' not in self.config['env'] or self.config['env']['data_dir'] is None:
                    self.config['env']['data_dir'] = data_path
                    # For dataset, use json_2.1.1 subdirectory if it exists
                    json_data_path = os.path.join(data_path, 'json_2.1.1')
                    if os.path.exists(json_data_path):
                        self.config['dataset']['data_path'] = json_data_path
                    else:
                        self.config['dataset']['data_path'] = data_path
                
                # Override environment type if specified
                if self.env_type:
                    self.config['env']['type'] = self.env_type
            
            # Fallback: if config loading failed, use manual config
            if not self.config:
                logger.warning("Failed to load ALFWorld config, using manual config")
                self.config = self._create_manual_config(data_path)
            
        except Exception as config_error:
            logger.warning(f"Error loading ALFWorld config: {config_error}")
            logger.info("Falling back to manual configuration")
            import os
            if self.data_dir:
                data_path = os.path.expanduser(self.data_dir)
            else:
                data_path = os.path.expanduser('~/.cache/alfworld')
            self.config = self._create_manual_config(data_path)
        
        # Ensure we have a config at this point
        if not self.config:
            logger.error("Failed to create ALFWorld config")
            raise RuntimeError("Could not create ALFWorld configuration")
        
        # Ensure data_path is set
        import os
        if 'env' not in self.config or 'data_dir' not in self.config['env']:
            data_path = os.path.expanduser('~/.cache/alfworld')
        else:
            data_path = self.config['env']['data_dir']
        
        # Check if data directory exists
        if not os.path.exists(data_path):
            logger.warning(
                f"ALFWorld data directory not found: {data_path}\n"
                f"Please run 'alfworld-download' to download ALFWorld data files."
            )
        else:
            logger.info(f"Using ALFWorld data directory: {data_path}")
        
        # Initialize environment with defensive config handling
        # Wrap in try-catch to provide defaults for any missing parameters
        logger.info(f"Creating ALFWorld environment of type: {self.config['env']['type']}")
        logger.debug(f"Config dataset path: {self.config.get('dataset', {}).get('data_path', 'not set')}")
        logger.debug(f"Config env data_dir: {self.config.get('env', {}).get('data_dir', 'not set')}")
        
        try:
            EnvClass = get_environment(self.config['env']['type'])
            logger.debug(f"Got environment class: {EnvClass}")
            logger.info("Instantiating ALFWorld environment...")
            env_instance = EnvClass(self.config, train_eval=self.train_eval)
            if env_instance is None:
                raise RuntimeError("EnvironClass instantiation returned None")
            logger.info(f"Environment instance created: {type(env_instance)}")
            
            logger.info("Initializing ALFWorld environment (this may take a moment)...")
            self.env = env_instance.init_env(batch_size=self.batch_size)
            logger.info(f"init_env returned: {type(self.env)}")
            
            if self.env is None:
                raise RuntimeError(
                    "ALFWorld init_env() returned None. This usually means:\n"
                    "1. No game files found in the data directory\n"
                    f"2. Check data path: {self.config.get('dataset', {}).get('data_path', 'not set')}\n"
                    "3. Run: alfworld-download --extra to download game files"
                )
            
            logger.info("ALFWorld environment initialization complete")
            # Only mark as initialized if env was successfully created
            if self.env is not None:
                self.is_initialized = True
                logger.info(f"ALFWorld environment initialized: {self.config['env']['type']}")
            else:
                raise RuntimeError("ALFWorld environment creation returned None")
                
        except KeyError as e:
            logger.error(
                f"ALFWorld configuration missing parameter: {e}\n"
                f"Current config keys: {list(self.config.keys()) if self.config else 'None'}\n"
                f"This suggests ALFWorld expects additional configuration parameters.\n"
                f"Full config structure:\n{self._config_to_string()}"
            )
            self.env = None
            self.is_initialized = False
            raise RuntimeError(
                f"ALFWorld configuration missing parameter: {e}\n"
                "See logs above for details."
            ) from e
        except Exception as e:
            logger.error("="*80)
            logger.error("ALFWorld INITIALIZATION FAILED")
            logger.error("="*80)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {e}")
            logger.exception("Full initialization traceback:")
            logger.error("="*80)
            
            self.env = None  # Ensure env is None on failure
            self.is_initialized = False  # Mark as not initialized
            
            # Provide specific guidance based on error
            error_str = str(e).lower()
            if "0 games" in error_str:
                logger.error(
                    "\n❌ ALFWorld found NO GAME FILES in split=train\n"
                    "This usually means:\n"
                    f"1. Data directory: {data_path}\n"
                    f"2. Dataset path: {self.config.get('dataset', {}).get('data_path', 'not set')}\n"
                    "3. ALFWorld expects TextWorld game files (.ulx/.z8), not just JSON data\n"
                    "4. Try running: alfworld-download --extra\n"
                    "5. Or check if TextWorld games need to be generated from PDDL files"
                )
            elif "keyerror" in error_str:
                logger.error(
                    "\n❌ Missing configuration parameter\n"
                    "The error above shows which parameter is missing.\n"
                    "Check the config file or add it to the defaults."
                )
            else:
                logger.error(
                    f"\n❌ Unknown initialization error: {e}\n"
                    "Check the full traceback above for details."
                )
            
            # Re-raise with context
            raise RuntimeError(
                f"ALFWorld initialization failed: {type(e).__name__}: {e}\n"
                "See logs above for detailed error information and troubleshooting steps."
            ) from e
    
    def _create_manual_config(self, data_path):
        """Create ALFWorld configuration manually as fallback."""
        return {
                'general': {
                    'training_method': 'dagger',  # Training method (dagger, behavior_cloning, etc.)
                    'seed': None,
                    'verbose': False
                },
                'env': {
                    'type': self.env_type or 'AlfredTWEnv',
                    'data_dir': data_path,
                    'num_processes': 1,
                    'goal_desc_human_anns_prob': 0.0,  # Probability of using human annotations
                    'goal_desc_human_anns_path': None,  # Path to human annotations
                    'save_period': 100,  # Save checkpoint every N episodes
                    'debug': False,
                    'seed': None,
                    'max_fails': 10,  # Max consecutive failures
                    'max_steps': 500,  # Max steps per episode
                    'reward_shaping': True,
                    'progress_type': 'reward',
                    'verbose': False,
                    'task_types': ['pick_and_place_simple', 'pick_clean_then_place_in_receiver',
                                   'pick_two_obj_and_place', 'look_at_obj_in_light', 
                                   'pick_and_place_with_movable_receiver'],
                    # Additional common parameters
                    'reward_type': 'dense',
                    'action_space': 'full',
                    'forward_model_applied': True,
                    'forward_model_max_steps': 5,
                    'randomize': True,
                    'domain_randomization': False,  # Domain randomization for training
                    'expert_type': 'human'  # Expert type for demonstrations (human, planner, etc.)
                },
                'dataset': {
                    'data_path': data_path,
                    'num_train_games': 0,  # 0 means use all available games
                    'splits': {
                        'train': ['train'],
                        'valid_unseen': ['valid_unseen'],
                        'valid_seen': ['valid_seen'],
                        'eval': ['eval']
                    }
                },
                'dagger': {
                    'training': {
                        'max_nb_steps_per_episode': 50,
                        'nb_rollout_steps': 5,
                        'update_freq': 10,
                        'batch_size': 32,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-5
                    },
                    'expert': {
                        'type': 'human',
                        'path': None
                    }
                }
            }
        
        # Check if data directory exists
        import os
        if not os.path.exists(data_path):
            logger.warning(
                f"ALFWorld data directory not found: {data_path}\n"
                f"Please run 'alfworld-download' to download ALFWorld data files."
            )
        else:
            logger.info(f"Using ALFWorld data directory: {data_path}")
        
        # Initialize environment with defensive config handling
        # Wrap in try-catch to provide defaults for any missing parameters
        logger.info(f"Creating ALFWorld environment of type: {self.config['env']['type']}")
        logger.debug(f"Config dataset path: {self.config.get('dataset', {}).get('data_path', 'not set')}")
        logger.debug(f"Config env data_dir: {self.config.get('env', {}).get('data_dir', 'not set')}")
        
        try:
            EnvClass = get_environment(self.config['env']['type'])
            logger.debug(f"Got environment class: {EnvClass}")
            logger.info("Instantiating ALFWorld environment...")
            env_instance = EnvClass(self.config, train_eval=self.train_eval)
            if env_instance is None:
                raise RuntimeError("EnvironClass instantiation returned None")
            logger.info(f"Environment instance created: {type(env_instance)}")
            
            logger.info("Initializing ALFWorld environment (this may take a moment)...")
            self.env = env_instance.init_env(batch_size=self.batch_size)
            logger.info(f"init_env returned: {type(self.env)}")
            
            if self.env is None:
                raise RuntimeError(
                    "ALFWorld init_env() returned None. This usually means:\n"
                    "1. No game files found in the data directory\n"
                    f"2. Check data path: {self.config.get('dataset', {}).get('data_path', 'not set')}\n"
                    "3. Run: alfworld-download --extra to download game files"
                )
            
            logger.info("ALFWorld environment initialization complete")
        except KeyError as ke:
            # If we hit a missing key, add a sensible default and retry
            missing_key = str(ke).strip("'\"")
            logger.warning(f"Adding missing config parameter: {missing_key}")
            
            # Try to determine which section the key is in
            error_str = str(ke)
            if "'general'" in error_str or '"general"' in error_str:
                # Common defaults for general parameters
                defaults = {
                    'training_method': 'dagger',
                    'seed': None,
                    'verbose': False,
                    'debug': False
                }
                if missing_key in defaults:
                    if 'general' not in self.config:
                        self.config['general'] = {}
                    self.config['general'][missing_key] = defaults[missing_key]
                    # Retry initialization
                    self.env = EnvClass(self.config, train_eval=self.train_eval)
                    self.env = self.env.init_env(batch_size=self.batch_size)
                else:
                    raise
            elif "'env'" in error_str or '"env"' in error_str:
                # Common defaults for env parameters
                defaults = {
                    'expert_type': 'human',
                    'expert_path': None,
                    'load_expert': False,
                    'expert_demos': 0,
                    'expert_demos_path': None,
                    'model': None,
                    'checkpoint': None,
                    'test': False,
                    'eval_only': False
                }
                if missing_key in defaults:
                    self.config['env'][missing_key] = defaults[missing_key]
                    # Retry initialization
                    self.env = EnvClass(self.config, train_eval=self.train_eval)
                    self.env = self.env.init_env(batch_size=self.batch_size)
                else:
                    raise
            elif "'dataset'" in str(ke) or '"dataset"' in str(ke):
                # Common defaults for dataset parameters
                defaults = {
                    'num_valid_games': 0,
                    'num_test_games': 0,
                    'expert_demos_path': None
                }
                if missing_key in defaults:
                    self.config['dataset'][missing_key] = defaults[missing_key]
                    # Retry initialization
                    self.env = EnvClass(self.config, train_eval=self.train_eval)
                    self.env = self.env.init_env(batch_size=self.batch_size)
                else:
                    raise
            elif "'dagger'" in str(ke) or '"dagger"' in str(ke):
                # Handle dagger training config
                if 'dagger' not in self.config:
                    self.config['dagger'] = {
                        'training': {
                            'max_nb_steps_per_episode': 50,
                            'nb_rollout_steps': 5,
                            'update_freq': 10,
                            'batch_size': 32,
                            'learning_rate': 1e-4,
                            'weight_decay': 1e-5
                        },
                        'expert': {
                            'type': 'human',
                            'path': None
                        }
                    }
                elif "'training'" in str(ke) or '"training"' in str(ke):
                    # Missing training parameter
                    if 'training' not in self.config['dagger']:
                        self.config['dagger']['training'] = {}
                    training_defaults = {
                        'max_nb_steps_per_episode': 50,
                        'nb_rollout_steps': 5,
                        'update_freq': 10,
                        'batch_size': 32,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-5
                    }
                    if missing_key in training_defaults:
                        self.config['dagger']['training'][missing_key] = training_defaults[missing_key]
                    else:
                        # Add a generic default
                        self.config['dagger']['training'][missing_key] = None
                # Retry initialization
                self.env = EnvClass(self.config, train_eval=self.train_eval)
                self.env = self.env.init_env(batch_size=self.batch_size)
            else:
                raise
            
            # Only mark as initialized if env was successfully created
            if self.env is not None:
                self.is_initialized = True
                logger.info(f"ALFWorld environment initialized: {self.config['env']['type']}")
            else:
                raise RuntimeError("ALFWorld environment creation returned None")
            
        except KeyError as e:
            logger.error(
                f"ALFWorld configuration missing parameter: {e}\n"
                f"Current config keys: {list(self.config.keys()) if self.config else 'None'}\n"
                f"This suggests ALFWorld expects additional configuration parameters.\n"
                f"Full config structure:\n{self._config_to_string()}"
            )
            self.env = None
            self.is_initialized = False
            raise RuntimeError(
                f"ALFWorld configuration missing parameter: {e}\n"
                "See logs above for details."
            ) from e
        except Exception as e:
            logger.error("="*80)
            logger.error("ALFWorld INITIALIZATION FAILED")
            logger.error("="*80)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {e}")
            logger.exception("Full initialization traceback:")
            logger.error("="*80)
            
            self.env = None  # Ensure env is None on failure
            self.is_initialized = False  # Mark as not initialized
            
            # Provide specific guidance based on error
            error_str = str(e).lower()
            if "0 games" in error_str:
                logger.error(
                    "\n❌ ALFWorld found NO GAME FILES in split=train\n"
                    "This usually means:\n"
                    f"1. Data directory: {data_path}\n"
                    f"2. Dataset path: {self.config.get('dataset', {}).get('data_path', 'not set')}\n"
                    "3. ALFWorld expects TextWorld game files (.ulx/.z8), not just JSON data\n"
                    "4. Try running: alfworld-download --extra\n"
                    "5. Or check if TextWorld games need to be generated from PDDL files"
                )
            elif "keyerror" in error_str:
                logger.error(
                    "\n❌ Missing configuration parameter\n"
                    "The error above shows which parameter is missing.\n"
                    "Check the config file or add it to the defaults."
                )
            else:
                logger.error(
                    f"\n❌ Unknown initialization error: {e}\n"
                    "Check the full traceback above for details."
                )
            
            # Re-raise with context
            raise RuntimeError(
                f"ALFWorld initialization failed: {type(e).__name__}: {e}\n"
                "See logs above for detailed error information and troubleshooting steps."
            ) from e
    
    def _config_to_string(self) -> str:
        """Convert config to readable string for debugging."""
        import json
        try:
            return json.dumps(self.config, indent=2, default=str)
        except:
            return str(self.config)
    
    def stop(self):
        """Stop/clean up the ALFWorld environment."""
        if self.env:
            try:
                self.env.close()
            except:
                pass
        self.env = None
        self.is_initialized = False
        self.current_observation = None
        self.current_info = None
        self.done = False
    
    def reset(self, task: Task) -> WebState:
        """Reset environment and start a new task.
        
        Args:
            task: The task to initialize
            
        Returns:
            Initial state of the environment
        """
        if not self.is_initialized:
            logger.info("Environment not initialized, calling start()...")
            try:
                self.start()
                logger.info(f"start() completed. self.env = {self.env}, self.is_initialized = {self.is_initialized}")
            except Exception as start_error:
                logger.error("="*80)
                logger.error("EXCEPTION CAUGHT IN reset() WHEN CALLING start()")
                logger.error("="*80)
                logger.error(f"Exception type: {type(start_error).__name__}")
                logger.error(f"Exception message: {start_error}")
                logger.exception("Full traceback:")
                logger.error("="*80)
                raise RuntimeError(
                    f"ALFWorld environment failed to start: {type(start_error).__name__}: {start_error}\n"
                    "Make sure ALFWorld data is downloaded (run: alfworld-download)\n"
                    "Check the logs above for detailed error information."
                ) from start_error
        
        # Ensure environment was successfully initialized
        if self.env is None:
            logger.error("="*80)
            logger.error("CRITICAL: self.env is None after start() returned")
            logger.error(f"self.is_initialized = {self.is_initialized}")
            logger.error("="*80)
            raise RuntimeError(
                "ALFWorld environment is None after initialization. "
                "This means start() completed without creating the environment.\n"
                "Check the logs above - there should be initialization messages.\n"
                "If there are no error messages above, start() may have failed silently.\n"
                "Make sure ALFWorld data is downloaded (run: alfworld-download)"
            )
        
        self.current_task = task
        self.done = False
        self.current_score = 0.0
        
        try:
            # Reset ALFWorld environment
            # ALFWorld tasks are typically loaded from its own task system
            # For now, we'll reset and use the observation
            result = self.env.reset()
            
            # ALFWorld's reset() returns (observation, info) where observation can be nested
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
            
            # Handle batch dimension (ALFWorld uses batching/nested tuples)
            # Observation might be: (('text',),) or ['text'] or 'text'
            while isinstance(obs, (tuple, list)) and len(obs) > 0:
                obs = obs[0]
            
            # Ensure observation is a string
            if not isinstance(obs, str):
                obs = str(obs)
            
            # Handle info batch dimension
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            elif not isinstance(info, dict):
                info = {}
            
            self.current_observation = obs
            self.current_info = info
            
            logger.info(f"ALFWorld task reset: {task.task_id}")
            logger.debug(f"Initial observation: {obs[:200]}...")
            
        except Exception as e:
            logger.error(f"Failed to reset ALFWorld environment: {e}")
            if self.env is None:
                raise RuntimeError(
                    "ALFWorld environment is not initialized. Initialization likely failed. "
                    "Check previous error messages. Make sure ALFWorld data is downloaded: alfworld-download"
                )
            # Fallback: create a dummy observation if reset fails but env exists
            self.current_observation = "You are in an ALFWorld environment. " + task.goal
            self.current_info = {"admissible_commands": ["look around", "examine objects"]}
        
        return self.get_state()
    
    def get_state(self) -> WebState:
        """Get current state of the environment.
        
        Returns:
            Current WebState (adapted for text-based environment)
        """
        if not self.is_initialized or self.current_observation is None:
            raise RuntimeError("Environment not started or not reset. Call start() and reset() first.")
        
        # Extract available commands from info
        available_actions = []
        
        if self.current_info and 'admissible_commands' in self.current_info:
            # ALFWorld provides admissible commands
            for idx, cmd in enumerate(self.current_info['admissible_commands']):
                if isinstance(cmd, list):
                    # Handle batch dimension
                    cmd = cmd[0] if cmd else ""
                
                if cmd:
                    available_actions.append({
                        "type": "text_command",
                        "command": cmd,
                        "index": idx,
                        "text": cmd
                    })
        else:
            # Fallback: create generic actions
            available_actions = [
                {"type": "text_command", "command": "look", "index": 0, "text": "look"},
                {"type": "text_command", "command": "examine", "index": 1, "text": "examine"},
                {"type": "text_command", "command": "go to", "index": 2, "text": "go to"}
            ]
        
        # Build task context
        task_context = {
            "task_id": self.current_task.task_id if self.current_task else None,
            "task_type": self.current_task.task_type.value if self.current_task else "alfworld",
            "current_location": self._extract_location(self.current_observation),
            "observation_length": len(self.current_observation)
        }
        
        goal = self.current_task.goal if self.current_task else "No task assigned"
        
        # Use observation text as "DOM tree" equivalent
        dom_tree = self.current_observation[:2000]  # Truncate if too long
        
        return WebState(
            url=f"alfworld://{task_context.get('task_id', 'unknown')}",  # Use fake URL for compatibility
            dom_tree=dom_tree,
            task_context=task_context,
            goal=goal,
            available_actions=available_actions,
            timestamp=time.time()
        )
    
    def _extract_location(self, observation: str) -> str:
        """Extract location information from observation."""
        # Simple heuristic: look for "You are in" or "You arrive at"
        lines = observation.split('\n')
        for line in lines[:5]:  # Check first few lines
            if "You are in" in line or "You arrive at" in line:
                return line.strip()
        return "unknown location"
    
    def execute_action(self, action: Dict[str, Any]) -> Tuple[WebState, Dict[str, Any]]:
        """Execute an action and return the new state.
        
        Args:
            action: Action dictionary with 'type' and 'command' fields
            
        Returns:
            Tuple of (next_state, outcome)
        """
        if not self.is_initialized:
            raise RuntimeError("Environment not started.")
        
        if self.done:
            logger.warning("Task already completed, ignoring action")
            return self.get_state(), {"success": False, "error": "Task already completed"}
        
        action_type = action.get("type", "text_command")
        outcome = {"success": False, "error": None, "reward": 0.0}
        
        try:
            if action_type == "text_command":
                # Extract command from action
                command = action.get("command") or action.get("text", "")
                
                if not command:
                    outcome["error"] = "No command provided"
                    return self.get_state(), outcome
                
                logger.debug(f"Executing ALFWorld command: {command}")
                
                # Step through environment
                # ALFWorld expects a list of actions (batch dimension)
                step_result = self.env.step([command])
                
                # step() returns (obs, scores, dones, infos) where each can be nested
                if isinstance(step_result, tuple) and len(step_result) >= 4:
                    obs, scores, dones, infos = step_result[:4]
                else:
                    obs, scores, dones, infos = step_result[0] if step_result else ("", None, False, {}), None, False, {}
                
                # Handle batch dimension (unwrap nested tuples/lists)
                while isinstance(obs, (tuple, list)) and len(obs) > 0:
                    obs = obs[0]
                
                # Ensure observation is a string
                if not isinstance(obs, str):
                    obs = str(obs)
                
                # Handle other return values
                while isinstance(scores, (tuple, list)) and len(scores) > 0:
                    scores = scores[0]
                while isinstance(dones, (tuple, list)) and len(dones) > 0:
                    dones = dones[0]
                if isinstance(infos, list) and len(infos) > 0:
                    infos = infos[0]
                elif not isinstance(infos, dict):
                    infos = {}
                
                self.current_observation = obs
                self.current_info = infos
                self.current_score = float(scores) if scores is not None else 0.0
                self.done = bool(dones) if dones is not None else False
                
                outcome["success"] = True
                outcome["reward"] = self.current_score
                outcome["done"] = self.done
                outcome["observation"] = obs[:200]  # Include snippet in outcome
                
                logger.debug(f"Action executed. Score: {self.current_score}, Done: {self.done}")
                
            else:
                outcome["error"] = f"Unknown action type: {action_type}"
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            outcome["error"] = str(e)
        
        next_state = self.get_state()
        return next_state, outcome
    
    def check_task_completion(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if the current task is complete.
        
        Returns:
            Tuple of (is_complete, completion_info)
        """
        if not self.is_initialized:
            return False, {}
        
        completion_info = {
            "done": self.done,
            "score": self.current_score,
            "observation": self.current_observation[:200] if self.current_observation else ""
        }
        
        # ALFWorld tasks are complete when done=True
        # Score > 0 typically indicates successful completion
        is_complete = self.done and self.current_score > 0
        
        return is_complete, completion_info

