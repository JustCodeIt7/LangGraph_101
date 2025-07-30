"""LangGraph nodes for input processing."""

from typing import Dict, Any, Optional, TypedDict
from ..input.handler import InputHandler, ProcessedInput
from ..input.config import InputConfig
from ..langgraph.nodes import RepositoryAnalysisState


class InputProcessorNode:
    """LangGraph node for input processing."""
    
    def __init__(self, input_config: Optional[InputConfig] = None):
        """Initialize the InputProcessorNode.
        
        Args:
            input_config: Input configuration, uses default if None
        """
        self.input_config = input_config or InputConfig()
        self.input_handler = InputHandler(self.input_config)
    
    def __call__(self, state: RepositoryAnalysisState) -> RepositoryAnalysisState:
        """Execute the input processing node.
        
        Args:
            state: Current workflow state containing input source
            
        Returns:
            Updated workflow state with processed input
        """
        try:
            # Extract repository source from state
            source = state.get("repository_url") or state.get("local_path")
            if not source:
                return {
                    **state,
                    "errors": state.get("errors", []) + ["No input source provided"],
                    "current_step": "input_processing_failed"
                }
            
            # Process input
            processed_input = self.input_handler.process(source)
            
            # Update state with processed input
            return {
                **state,
                "processed_input": processed_input,
                "local_path": processed_input.local_path,
                "current_step": "input_processed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Input processing failed: {str(e)}"],
                "current_step": "input_processing_failed"
            }