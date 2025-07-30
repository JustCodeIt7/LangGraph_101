"""LangGraph nodes for README generation."""

from typing import Dict, Any, Optional
from .generator import READMEGenerator
from ..langgraph.nodes import RepositoryAnalysisState


class READMEGeneratorNode:
    """LangGraph node for README generation from repository analysis."""
    
    def __init__(self):
        """Initialize the READMEGeneratorNode."""
        self.generator = READMEGenerator()
    
    def __call__(self, state: RepositoryAnalysisState) -> RepositoryAnalysisState:
        """Execute the README generation node.
        
        Args:
            state: Current workflow state containing analysis results
            
        Returns:
            Updated workflow state with generated README content
        """
        try:
            # Extract repository structure from state
            structure = state.get("repository_structure")
            if not structure:
                return {
                    **state,
                    "errors": state.get("errors", []) + ["No repository structure provided for README generation"],
                    "current_step": "readme_generation_failed"
                }
            
            # Generate README content
            readme_content = self.generator.generate_readme(structure)
            
            # Update state with generated README
            return {
                **state,
                "generated_readme": readme_content,
                "current_step": "readme_generation_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"README generation failed: {str(e)}"],
                "current_step": "readme_generation_failed"
            }


def get_readme_generator_node() -> READMEGeneratorNode:
    """Get a configured README generator node.
    
    Returns:
        READMEGeneratorNode instance
    """
    return READMEGeneratorNode()


def add_readme_generation_to_workflow(workflow, config: Optional[Any] = None) -> None:
    """Add README generation node to an existing workflow.
    
    Args:
        workflow: LangGraph workflow to modify
        config: Configuration (reserved for future use)
    """
    # Add README generation node
    readme_node = READMEGeneratorNode()
    workflow.add_node("generate_readme", readme_node)
    
    # Connect README generation after analysis
    workflow.add_edge("analyze_repository", "generate_readme")