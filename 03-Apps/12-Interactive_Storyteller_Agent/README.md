# Interactive Storyteller Agent

## Description
This agent creates interactive "Choose Your Own Adventure" style narratives. It uses state-driven branching dialogues and tracks user choices to maintain story continuity.

## Pseudocode
```
FUNCTION start_story(story_id, user_session)
  story_data = LOAD_STORY_DATA(story_id)
  user_session.current_node_id = story_data.start_node_id
  user_session.story_state = INITIALIZE_STORY_STATE(story_data.initial_variables)
  user_session.history = [] // To track choices

  current_node = GET_NODE_BY_ID(story_data, user_session.current_node_id)
  processed_text = PROCESS_NODE_TEXT(current_node.text, user_session.story_state)
  available_choices = GET_AVAILABLE_CHOICES(current_node, user_session.story_state)

  SAVE_USER_SESSION(user_session)
  RETURN CREATE_RESPONSE(processed_text, available_choices)
END FUNCTION

FUNCTION handle_user_choice(user_session_id, choice_id)
  user_session = LOAD_USER_SESSION(user_session_id)
  story_data = LOAD_STORY_DATA(user_session.story_id) // Assuming story_id is part of session

  current_node = GET_NODE_BY_ID(story_data, user_session.current_node_id)
  chosen_option = GET_OPTION_BY_ID(current_node.choices, choice_id)

  IF chosen_option IS NULL OR NOT IS_CHOICE_VALID(chosen_option, user_session.story_state) THEN
    // Handle invalid choice, perhaps repeat current node's text and choices
    processed_text = PROCESS_NODE_TEXT(current_node.text, user_session.story_state)
    available_choices = GET_AVAILABLE_CHOICES(current_node, user_session.story_state)
    RETURN CREATE_RESPONSE("Invalid choice. " + processed_text, available_choices)
  ENDIF

  // Update story state based on choice effects
  IF chosen_option.effects IS NOT NULL THEN
    APPLY_EFFECTS_TO_STATE(user_session.story_state, chosen_option.effects)
  ENDIF

  ADD_TO_HISTORY(user_session.history, user_session.current_node_id, choice_id)
  user_session.current_node_id = chosen_option.next_node_id

  next_node = GET_NODE_BY_ID(story_data, user_session.current_node_id)
  processed_text = PROCESS_NODE_TEXT(next_node.text, user_session.story_state)
  available_choices = GET_AVAILABLE_CHOICES(next_node, user_session.story_state)

  IF IS_END_NODE(next_node) THEN
    // Handle story ending
    processed_text += "\n\n--- THE END ---"
    CLEAR_USER_SESSION(user_session_id) // Or mark as completed
  ELSE
    SAVE_USER_SESSION(user_session)
  ENDIF

  RETURN CREATE_RESPONSE(processed_text, available_choices)
END FUNCTION

FUNCTION LOAD_STORY_DATA(story_id)
  // Load from JSON, XML, database, etc.
  // Story data includes: nodes, choices, conditions, effects, initial variables
  // Example Node: { id: "node1", text: "You are at a crossroads. Go left or right?", choices: [{id:"c1", text:"Go left", next_node_id:"node2"}, {id:"c2", text:"Go right", next_node_id:"node3"}] }
  // Example Choice with condition/effect: {id:"c3", text:"Open chest (requires key)", next_node_id:"node4", condition:"story_state.has_key == true", effects: {"story_state.gold += 10"}}
  RETURN GET_STORY_FILE_CONTENT(story_id + ".json")
END FUNCTION

FUNCTION PROCESS_NODE_TEXT(text, story_state)
  // Replace placeholders in text with values from story_state
  // e.g., "Hello {story_state.player_name}, you have {story_state.gold} gold."
  return RENDER_TEMPLATE_STRING(text, story_state)
END FUNCTION

FUNCTION GET_AVAILABLE_CHOICES(node, story_state)
  // Filter choices based on conditions in story_state
  valid_choices = []
  FOR EACH choice IN node.choices
    IF choice.condition IS NULL OR EVALUATE_CONDITION(choice.condition, story_state) THEN
      ADD_TO_LIST(valid_choices, choice)
    ENDIF
  ENDFOR
  RETURN valid_choices
END FUNCTION

FUNCTION EVALUATE_CONDITION(condition_string, story_state)
  // Safely evaluate a condition string like "story_state.has_key == true && story_state.strength > 5"
  // This needs a secure expression evaluator
  RETURN SAFE_EVAL(condition_string, {"story_state": story_state})
END FUNCTION

FUNCTION APPLY_EFFECTS_TO_STATE(story_state, effects_map)
  // Modify story_state based on effects, e.g., {"gold += 10", "has_key = false"}
  FOR EACH variable, operation IN effects_map
    // Safely apply operation to story_state.variable
    // e.g., story_state["gold"] = story_state["gold"] + 10
    EXECUTE_STATE_MODIFICATION(story_state, variable, operation)
  ENDFOR
END FUNCTION

FUNCTION CREATE_RESPONSE(text, choices)
  // Format for display to user
  response_object = { "narrative_text": text, "options": [] }
  FOR EACH choice IN choices
    ADD_TO_LIST(response_object.options, { "id": choice.id, "text": choice.text })
  ENDFOR
  RETURN response_object
END FUNCTION
```
