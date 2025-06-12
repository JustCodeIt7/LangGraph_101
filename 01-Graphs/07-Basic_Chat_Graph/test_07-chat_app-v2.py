import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_app_v2 import (
    ChatState,
    initialize_state,
    process_user_input,
    generate_ai_response,
    should_continue,
    create_llm
)


class TestChatState:
    """Test the ChatState TypedDict structure."""
    
    def test_chat_state_structure(self):
        """Test that ChatState has all required fields."""
        state: ChatState = {
            "messages": [],
            "current_response": "",
            "exit_requested": False,
            "verbose_mode": False,
            "command_processed": False
        }
        
        assert isinstance(state["messages"], list)
        assert isinstance(state["current_response"], str)
        assert isinstance(state["exit_requested"], bool)
        assert isinstance(state["verbose_mode"], bool)
        assert isinstance(state["command_processed"], bool)


class TestInitializeState:
    """Test the initialize_state function."""
    
    def test_initialize_state(self):
        """Test that state is properly initialized."""
        state: ChatState = {}
        result = initialize_state(state)
        
        assert result["messages"] == []
        assert result["current_response"] == ""
        assert result["exit_requested"] is False
        assert result["verbose_mode"] is False
        assert result["command_processed"] is False


class TestProcessUserInput:
    """Test the process_user_input function."""
    
    def setup_method(self):
        """Set up test state."""
        self.state: ChatState = {
            "messages": [],
            "current_response": "",
            "exit_requested": False,
            "verbose_mode": False,
            "command_processed": False
        }
    
    @patch('chat_app_v2.Prompt.ask')
    def test_exit_commands(self, mock_prompt):
        """Test exit commands."""
        exit_commands = ["exit", "quit", "bye", "EXIT", "QUIT", "BYE"]
        
        for command in exit_commands:
            mock_prompt.return_value = command
            result = process_user_input(self.state.copy())
            
            assert result["exit_requested"] is True
            assert result["command_processed"] is True
    
    @patch('chat_app_v2.Prompt.ask')
    @patch('chat_app_v2.console.print')
    def test_verbose_toggle(self, mock_print, mock_prompt):
        """Test verbose mode toggle."""
        mock_prompt.return_value = "verbose"
        
        # First toggle - should enable verbose
        result = process_user_input(self.state.copy())
        assert result["verbose_mode"] is True
        assert result["command_processed"] is True
        
        # Second toggle - should disable verbose
        state_with_verbose = result.copy()
        result2 = process_user_input(state_with_verbose)
        assert result2["verbose_mode"] is False
        assert result2["command_processed"] is True
    
    @patch('chat_app_v2.Prompt.ask')
    @patch('chat_app_v2.console.print')
    def test_help_command(self, mock_print, mock_prompt):
        """Test help command."""
        help_commands = ["help", "?", "HELP"]
        
        for command in help_commands:
            mock_prompt.return_value = command
            result = process_user_input(self.state.copy())
            
            assert result["command_processed"] is True
            assert mock_print.called
    
    @patch('chat_app_v2.Prompt.ask')
    def test_regular_message(self, mock_prompt):
        """Test regular user message processing."""
        mock_prompt.return_value = "Hello, how are you?"
        
        result = process_user_input(self.state.copy())
        
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, how are you?"
        assert result["command_processed"] is False
    
    @patch('chat_app_v2.Prompt.ask')
    def test_empty_message(self, mock_prompt):
        """Test empty message handling."""
        mock_prompt.return_value = "   "
        
        result = process_user_input(self.state.copy())
        
        assert len(result["messages"]) == 0
        assert result["command_processed"] is False
    
    @patch('chat_app_v2.Prompt.ask')
    @patch('chat_app_v2.console.print')
    def test_keyboard_interrupt(self, mock_print, mock_prompt):
        """Test keyboard interrupt handling."""
        mock_prompt.side_effect = KeyboardInterrupt()
        
        result = process_user_input(self.state.copy())
        
        assert result["exit_requested"] is True
        assert result["command_processed"] is True
    
    @patch('chat_app_v2.Prompt.ask')
    @patch('chat_app_v2.console.print')
    def test_exception_handling(self, mock_print, mock_prompt):
        """Test general exception handling."""
        mock_prompt.side_effect = Exception("Test error")
        
        result = process_user_input(self.state.copy())
        
        assert result["command_processed"] is True
        assert mock_print.called


class TestGenerateAIResponse:
    """Test the generate_ai_response function."""
    
    def setup_method(self):
        """Set up test state and mock LLM."""
        self.state: ChatState = {
            "messages": [{"role": "user", "content": "Hello"}],
            "current_response": "",
            "exit_requested": False,
            "verbose_mode": False,
            "command_processed": False
        }
        
        self.mock_llm = Mock()
        self.mock_response = Mock()
        self.mock_response.content = "Hello! How can I help you?"
        self.mock_llm.invoke.return_value = self.mock_response
    
    @patch('chat_app_v2.console.print')
    def test_successful_ai_response(self, mock_print):
        """Test successful AI response generation."""
        result = generate_ai_response(self.state.copy(), self.mock_llm)
        
        assert len(result["messages"]) == 2
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "Hello! How can I help you?"
        assert result["current_response"] == "Hello! How can I help you?"
        assert self.mock_llm.invoke.called
    
    def test_skip_on_exit_requested(self):
        """Test that AI response is skipped when exit is requested."""
        state = self.state.copy()
        state["exit_requested"] = True
        
        result = generate_ai_response(state, self.mock_llm)
        
        assert not self.mock_llm.invoke.called
        assert len(result["messages"]) == 1  # No new message added
    
    def test_skip_on_command_processed(self):
        """Test that AI response is skipped when command was processed."""
        state = self.state.copy()
        state["command_processed"] = True
        
        result = generate_ai_response(state, self.mock_llm)
        
        assert not self.mock_llm.invoke.called
        assert len(result["messages"]) == 1  # No new message added
    
    def test_skip_on_empty_messages(self):
        """Test that AI response is skipped when no messages exist."""
        state = self.state.copy()
        state["messages"] = []
        
        result = generate_ai_response(state, self.mock_llm)
        
        assert not self.mock_llm.invoke.called
        assert len(result["messages"]) == 0
    
    def test_skip_on_non_user_last_message(self):
        """Test that AI response is skipped when last message is not from user."""
        state = self.state.copy()
        state["messages"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = generate_ai_response(state, self.mock_llm)
        
        assert not self.mock_llm.invoke.called
        assert len(result["messages"]) == 2  # No new message added
    
    @patch('chat_app_v2.console.print')
    def test_verbose_mode_output(self, mock_print):
        """Test verbose mode output."""
        state = self.state.copy()
        state["verbose_mode"] = True
        
        result = generate_ai_response(state, self.mock_llm)
        
        # Check that verbose information was printed
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("Debug Info:" in str(call) for call in print_calls)


class TestShouldContinue:
    """Test the should_continue function."""
    
    def setup_method(self):
        """Set up test state."""
        self.state: ChatState = {
            "messages": [],
            "current_response": "",
            "exit_requested": False,
            "verbose_mode": False,
            "command_processed": False
        }
    
    @patch('chat_app_v2.console.print')
    def test_exit_requested(self, mock_print):
        """Test when exit is requested."""
        state = self.state.copy()
        state["exit_requested"] = True
        
        result = should_continue(state)
        
        assert result == "end"
        assert mock_print.called
    
    def test_command_processed(self):
        """Test when command was processed."""
        state = self.state.copy()
        state["command_processed"] = True
        
        result = should_continue(state)
        
        assert result == "continue_input"
    
    def test_normal_flow(self):
        """Test normal conversation flow."""
        result = should_continue(self.state)
        
        assert result == "continue_input"


class TestCreateLLM:
    """Test the create_llm function."""
    
    @patch('chat_app_v2.ChatLiteLLM')
    @patch('chat_app_v2.console.print')
    def test_create_llm(self, mock_print, mock_chat_lite_llm):
        """Test LLM creation."""
        mock_llm_instance = Mock()
        mock_chat_lite_llm.return_value = mock_llm_instance
        
        result = create_llm()
        
        assert result == mock_llm_instance
        assert mock_chat_lite_llm.called
        
        # Verify the LLM was created with correct parameters
        call_args = mock_chat_lite_llm.call_args
        assert call_args[1]["model"] == "ollama/qwen3:0.6b"
        assert call_args[1]["api_base"] == "http://localhost:11434"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 1000


class TestIntegration:
    """Integration tests for the chat application."""
    
    @patch('chat_app_v2.Prompt.ask')
    @patch('chat_app_v2.console.print')
    def test_conversation_flow(self, mock_print, mock_prompt):
        """Test a complete conversation flow."""
        # Simulate user saying hello and then exiting
        mock_prompt.side_effect = ["Hello", "exit"]
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Hi there! How can I help you?"
        mock_llm.invoke.return_value = mock_response
        
        # Initialize state
        state = initialize_state({})
        
        # Process first user input
        state = process_user_input(state)
        assert len(state["messages"]) == 1
        assert not state["exit_requested"]
        
        # Generate AI response
        state = generate_ai_response(state, mock_llm)
        assert len(state["messages"]) == 2
        
        # Process exit command
        state = process_user_input(state)
        assert state["exit_requested"]
        assert state["command_processed"]
        
        # Check flow control
        result = should_continue(state)
        assert result == "end"