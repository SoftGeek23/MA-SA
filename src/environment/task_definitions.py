"""Task type definitions for deterministic web tasks."""
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class TaskType(Enum):
    """Types of web tasks."""
    SEARCH_CLICK = "search_click"
    FORM_FILL = "form_fill"
    FILTER_EXTRACT = "filter_extract"
    SORT_SELECT = "sort_select"
    NAV_TOGGLE = "nav_toggle"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class Task:
    """Represents a web task."""
    task_id: str
    task_type: TaskType
    description: str
    goal: str
    url: str
    success_criteria: Dict[str, Any]
    initial_state: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate task."""
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {self.url}")


class TaskDefinition:
    """Factory for creating task instances."""
    
    @staticmethod
    def create_search_click_task(
        task_id: str,
        description: str,
        url: str,
        search_query: str,
        target_element: str
    ) -> Task:
        """Create a search+click task."""
        goal = f"Search for '{search_query}' and click on '{target_element}'"
        success_criteria = {
            "type": "element_clicked",
            "element_selector": target_element,
            "page_contains": search_query
        }
        return Task(
            task_id=task_id,
            task_type=TaskType.SEARCH_CLICK,
            description=description,
            goal=goal,
            url=url,
            success_criteria=success_criteria
        )
    
    @staticmethod
    def create_form_fill_task(
        task_id: str,
        description: str,
        url: str,
        form_fields: Dict[str, str],
        submit_button: str
    ) -> Task:
        """Create a form-fill task."""
        fields_str = ", ".join([f"{k}={v}" for k, v in form_fields.items()])
        goal = f"Fill form with: {fields_str} and submit"
        success_criteria = {
            "type": "form_submitted",
            "fields_filled": list(form_fields.keys()),
            "submit_button": submit_button
        }
        return Task(
            task_id=task_id,
            task_type=TaskType.FORM_FILL,
            description=description,
            goal=goal,
            url=url,
            success_criteria=success_criteria
        )
    
    @staticmethod
    def create_filter_extract_task(
        task_id: str,
        description: str,
        url: str,
        filter_criteria: Dict[str, Any],
        extract_target: str
    ) -> Task:
        """Create a filter+extract task."""
        filter_str = ", ".join([f"{k}={v}" for k, v in filter_criteria.items()])
        goal = f"Apply filters: {filter_str} and extract '{extract_target}'"
        success_criteria = {
            "type": "data_extracted",
            "filters_applied": filter_criteria,
            "extract_target": extract_target
        }
        return Task(
            task_id=task_id,
            task_type=TaskType.FILTER_EXTRACT,
            description=description,
            goal=goal,
            url=url,
            success_criteria=success_criteria
        )
    
    @staticmethod
    def create_sort_select_task(
        task_id: str,
        description: str,
        url: str,
        sort_by: str,
        select_criteria: str
    ) -> Task:
        """Create a sort+select task."""
        goal = f"Sort by '{sort_by}' and select items matching '{select_criteria}'"
        success_criteria = {
            "type": "items_selected",
            "sort_by": sort_by,
            "select_criteria": select_criteria
        }
        return Task(
            task_id=task_id,
            task_type=TaskType.SORT_SELECT,
            description=description,
            goal=goal,
            url=url,
            success_criteria=success_criteria
        )
    
    @staticmethod
    def create_nav_toggle_task(
        task_id: str,
        description: str,
        url: str,
        navigation_path: List[str]
    ) -> Task:
        """Create a navigation toggle task."""
        path_str = " -> ".join(navigation_path)
        goal = f"Navigate through: {path_str}"
        success_criteria = {
            "type": "navigation_complete",
            "path": navigation_path
        }
        return Task(
            task_id=task_id,
            task_type=TaskType.NAV_TOGGLE,
            description=description,
            goal=goal,
            url=url,
            success_criteria=success_criteria
        )
    
    @staticmethod
    def create_error_recovery_task(
        task_id: str,
        description: str,
        url: str,
        error_scenario: str,
        recovery_action: str
    ) -> Task:
        """Create an error recovery task."""
        goal = f"Encounter error: {error_scenario} and recover by: {recovery_action}"
        success_criteria = {
            "type": "error_recovered",
            "error_scenario": error_scenario,
            "recovery_action": recovery_action
        }
        return Task(
            task_id=task_id,
            task_type=TaskType.ERROR_RECOVERY,
            description=description,
            goal=goal,
            url=url,
            success_criteria=success_criteria
        )

