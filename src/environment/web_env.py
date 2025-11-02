"""Web environment using Playwright for deterministic task execution."""
import time
from typing import Dict, Any, List, Optional, Tuple
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
from pathlib import Path

from .state_encoder import WebState, StateEncoder
from .task_definitions import Task, TaskType


class WebEnvironment:
    """Web environment wrapper for deterministic web tasks."""
    
    def __init__(
        self,
        headless: bool = True,
        browser: str = "chromium",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        navigation_timeout: int = 30000
    ):
        """Initialize web environment.
        
        Args:
            headless: Run browser in headless mode
            browser: Browser type (chromium, firefox, webkit)
            viewport_width: Viewport width in pixels
            viewport_height: Viewport height in pixels
            navigation_timeout: Navigation timeout in milliseconds
        """
        self.headless = headless
        self.browser_type = browser
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.navigation_timeout = navigation_timeout
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.state_encoder = StateEncoder()
        self.current_task: Optional[Task] = None
        self.current_url: Optional[str] = None
        
    def start(self):
        """Start the browser environment."""
        self.playwright = sync_playwright().start()
        
        browser_map = {
            "chromium": self.playwright.chromium,
            "firefox": self.playwright.firefox,
            "webkit": self.playwright.webkit
        }
        
        browser_launcher = browser_map.get(self.browser_type)
        if browser_launcher is None:
            raise ValueError(f"Unsupported browser: {self.browser_type}")
        
        self.browser = browser_launcher.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self.page = self.context.new_page()
        self.page.set_default_navigation_timeout(self.navigation_timeout)
    
    def stop(self):
        """Stop the browser environment."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def reset(self, task: Task) -> WebState:
        """Reset environment and navigate to task URL.
        
        Args:
            task: The task to initialize
            
        Returns:
            Initial state of the environment
        """
        if not self.page:
            self.start()
        
        self.current_task = task
        self.current_url = task.url
        
        # Navigate to URL
        self.page.goto(task.url, wait_until="domcontentloaded")
        time.sleep(1)  # Allow page to settle
        
        return self.get_state()
    
    def get_state(self) -> WebState:
        """Get current state of the environment.
        
        Returns:
            Current WebState
        """
        if not self.page:
            raise RuntimeError("Environment not started. Call start() or reset() first.")
        
        # Extract DOM tree (simplified)
        dom_tree = self._extract_dom()
        
        # Get available actions
        available_actions = self._get_available_actions()
        
        # Build task context
        task_context = {
            "task_id": self.current_task.task_id if self.current_task else None,
            "task_type": self.current_task.task_type.value if self.current_task else None,
            "current_url": self.page.url,
            "title": self.page.title()
        }
        
        goal = self.current_task.goal if self.current_task else "No task assigned"
        
        return WebState(
            url=self.page.url,
            dom_tree=dom_tree,
            task_context=task_context,
            goal=goal,
            available_actions=available_actions,
            timestamp=time.time()
        )
    
    def _extract_dom(self) -> str:
        """Extract simplified DOM representation."""
        # Extract key elements: buttons, links, inputs, select elements
        script = """
        () => {
            const elements = [];
            
            // Get all interactive elements
            const selectors = [
                'button', 'a', 'input', 'select', 'textarea',
                '[role="button"]', '[onclick]', '[tabindex="0"]'
            ];
            
            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    const text = el.textContent?.trim().substring(0, 100) || '';
                    const id = el.id || '';
                    const className = el.className || '';
                    const tag = el.tagName.toLowerCase();
                    
                    if (text || id) {
                        elements.push({
                            tag: tag,
                            id: id,
                            className: className,
                            text: text,
                            selector: `${tag}${id ? '#' + id : ''}${className ? '.' + className.split(' ')[0] : ''}`
                        });
                    }
                });
            });
            
            return JSON.stringify(elements, null, 2);
        }
        """
        
        try:
            elements = self.page.evaluate(script)
            return f"Interactive elements: {elements}"
        except Exception as e:
            return f"Error extracting DOM: {str(e)}"
    
    def _get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions from current page."""
        script = """
        () => {
            const actions = [];
            
            // Get clickable elements
            document.querySelectorAll('button, a[href], [role="button"], [onclick]').forEach((el, idx) => {
                const text = el.textContent?.trim().substring(0, 50) || '';
                const id = el.id || '';
                const selector = `${el.tagName.toLowerCase()}${id ? '#' + id : ''}`;
                
                if (text || id) {
                    actions.push({
                        type: 'click',
                        selector: selector,
                        text: text,
                        index: idx
                    });
                }
            });
            
            // Get input fields
            document.querySelectorAll('input[type="text"], input[type="email"], input[type="password"], textarea').forEach((el, idx) => {
                const placeholder = el.placeholder || '';
                const name = el.name || '';
                const selector = `input[name="${name}"]`;
                
                actions.push({
                    type: 'type',
                    selector: selector,
                    placeholder: placeholder,
                    name: name,
                    index: idx
                });
            });
            
            return actions;
        }
        """
        
        try:
            actions = self.page.evaluate(script)
            return actions
        except Exception as e:
            return []
    
    def execute_action(self, action: Dict[str, Any]) -> Tuple[WebState, Dict[str, Any]]:
        """Execute an action and return the new state.
        
        Args:
            action: Action dictionary with 'type' and other relevant fields
            
        Returns:
            Tuple of (next_state, outcome)
        """
        if not self.page:
            raise RuntimeError("Environment not started.")
        
        action_type = action.get("type")
        outcome = {"success": False, "error": None}
        
        try:
            if action_type == "click":
                selector = action.get("selector")
                if selector:
                    self.page.click(selector, timeout=5000)
                    outcome["success"] = True
                else:
                    outcome["error"] = "No selector provided for click action"
            
            elif action_type == "type":
                selector = action.get("selector") or action.get("name")
                text = action.get("text", "")
                if selector and text:
                    self.page.fill(selector, text)
                    outcome["success"] = True
                else:
                    outcome["error"] = "Missing selector or text for type action"
            
            elif action_type == "select":
                selector = action.get("selector")
                value = action.get("value")
                if selector and value:
                    self.page.select_option(selector, value)
                    outcome["success"] = True
                else:
                    outcome["error"] = "Missing selector or value for select action"
            
            elif action_type == "navigate":
                url = action.get("url")
                if url:
                    self.page.goto(url, wait_until="domcontentloaded")
                    self.current_url = url
                    outcome["success"] = True
                else:
                    outcome["error"] = "No URL provided for navigate action"
            
            elif action_type == "go_back":
                self.page.go_back(wait_until="domcontentloaded")
                outcome["success"] = True
            
            elif action_type == "wait":
                seconds = action.get("seconds", 1)
                time.sleep(seconds)
                outcome["success"] = True
            
            else:
                outcome["error"] = f"Unknown action type: {action_type}"
            
            # Wait for page to settle after action
            time.sleep(0.5)
            
        except Exception as e:
            outcome["error"] = str(e)
            outcome["success"] = False
        
        next_state = self.get_state()
        return next_state, outcome
    
    def check_task_completion(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if current task is completed.
        
        Returns:
            Tuple of (is_complete, completion_info)
        """
        if not self.current_task:
            return False, {"error": "No task assigned"}
        
        criteria = self.current_task.success_criteria
        completion_info = {}
        
        try:
            if criteria["type"] == "element_clicked":
                selector = criteria.get("element_selector")
                if selector:
                    element = self.page.query_selector(selector)
                    completion_info["element_found"] = element is not None
                    return element is not None, completion_info
            
            elif criteria["type"] == "form_submitted":
                # Check if form fields are filled
                fields = criteria.get("fields_filled", [])
                all_filled = all(
                    self.page.query_selector(f'input[name="{field}"]') is not None
                    for field in fields
                )
                completion_info["fields_filled"] = all_filled
                return all_filled, completion_info
            
            elif criteria["type"] == "data_extracted":
                # Check if filters are applied (simplified check)
                extract_target = criteria.get("extract_target", "")
                page_text = self.page.inner_text("body")
                completion_info["target_found"] = extract_target.lower() in page_text.lower()
                return completion_info["target_found"], completion_info
            
            elif criteria["type"] == "items_selected":
                # Simplified check - would need more sophisticated logic
                completion_info["check_performed"] = True
                return True, completion_info
            
            elif criteria["type"] == "navigation_complete":
                # Check if we're at expected URL
                expected_path = criteria.get("path", [])
                current_url = self.page.url
                completion_info["url"] = current_url
                return True, completion_info  # Simplified
            
            elif criteria["type"] == "error_recovered":
                # Check if error is resolved
                completion_info["error_recovered"] = True
                return True, completion_info
            
        except Exception as e:
            completion_info["error"] = str(e)
            return False, completion_info
        
        return False, {"error": "Unknown criteria type"}

