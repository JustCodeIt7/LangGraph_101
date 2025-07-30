"""LangGraph nodes for repository analysis."""

from typing import Dict, Any, Optional, TypedDict
from langgraph import StateGraph, END
from ..core.analyzer import RepositoryAnalyzer
from ..core.config import AnalysisConfig
from ..core.data_structures import RepositoryStructure


class RepositoryAnalysisState(TypedDict):
    """State schema for repository analysis workflows."""
    # Input
    repository_url: str
    local_path: Optional[str]
    
    # Configuration
    analysis_config: Optional[AnalysisConfig]
    
    # Analysis Results
    repository_structure: Optional[RepositoryStructure]
    analysis_summary: Optional[Dict[str, Any]]
    generated_readme: Optional[str]
    
    # Process Management
    current_step: str
    errors: list[str]
    warnings: list[str]


class RepositoryAnalyzerNode:
    """LangGraph node for repository structure analysis."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the RepositoryAnalyzerNode.
        
        Args:
            config: Analysis configuration, uses default if None
        """
        self.config = config
        self.analyzer = RepositoryAnalyzer(config)
    
    def __call__(self, state: RepositoryAnalysisState) -> RepositoryAnalysisState:
        """Execute the repository analysis node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            # Extract repository source from state
            repo_source = state.get("repository_url") or state.get("local_path")
            if not repo_source:
                return {
                    **state,
                    "errors": state.get("errors", []) + ["No repository source provided"],
                    "current_step": "analysis_failed"
                }
            
            # Perform analysis
            structure = self.analyzer.analyze(repo_source)
            
            # Create analysis summary
            summary = self._create_analysis_summary(structure)
            
            # Update state
            return {
                **state,
                "repository_structure": structure,
                "analysis_summary": summary,
                "current_step": "analysis_complete"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [str(e)],
                "current_step": "analysis_failed"
            }
        finally:
            # Clean up temporary resources
            self.analyzer.cleanup()
    
    def _create_analysis_summary(self, structure: RepositoryStructure) -> Dict[str, Any]:
        """Create a summary of the analysis results.
        
        Args:
            structure: RepositoryStructure object with analysis results
            
        Returns:
            Dictionary containing analysis summary
        """
        return {
            "project_name": structure.metadata.name,
            "primary_language": structure.metadata.primary_language,
            "languages": structure.metadata.languages,
            "frameworks": structure.metadata.frameworks,
            "architecture_type": structure.metadata.architecture_type,
            "complexity_score": structure.metadata.complexity_score,
            "documentation_coverage": structure.metadata.documentation_coverage,
            "test_coverage_estimate": structure.metadata.test_coverage_estimate,
            "entry_points": structure.metadata.entry_points,
            "configuration_files": structure.metadata.configuration_files,
            "total_files": len(structure.files),
            "total_directories": len(structure.directories),
            "detected_frameworks": len(structure.frameworks),
            "detected_patterns": len(structure.patterns),
            "mapped_relationships": len(structure.relationships)
        }


def get_analyzer_node(config: Optional[AnalysisConfig] = None) -> RepositoryAnalyzerNode:
    """Get a configured repository analyzer node.
    
    Args:
        config: Analysis configuration, uses default if None
        
    Returns:
        RepositoryAnalyzerNode instance
    """
    return RepositoryAnalyzerNode(config)


def create_analysis_workflow(config: Optional[AnalysisConfig] = None) -> StateGraph:
    """Create a complete LangGraph workflow for repository analysis.
    
    Args:
        config: Analysis configuration, uses default if None
        
    Returns:
        Configured StateGraph workflow
    """
    from ..readme_gen.nodes import READMEGeneratorNode
    
    workflow = StateGraph(RepositoryAnalysisState)
    
    # Add nodes
    analyzer_node = RepositoryAnalyzerNode(config)
    readme_node = READMEGeneratorNode()
    
    workflow.add_node("analyze_repository", analyzer_node)
    workflow.add_node("generate_readme", readme_node)
    
    # Define edges
    workflow.add_edge("analyze_repository", "generate_readme")
    workflow.add_edge("generate_readme", END)
    
    # Set entry point
    workflow.set_entry_point("analyze_repository")
    
    return workflow.compile()