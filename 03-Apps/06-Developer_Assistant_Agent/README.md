# Developer Assistant Agent

## Description
This agent assists developers with coding tasks. It can generate code from natural language specifications, perform automated code review, linting, and vulnerability checks, and provide in-line explanations of generated code.

## Pseudocode
```
FUNCTION handle_developer_request(user_request, current_codebase_context)
  intent = IDENTIFY_INTENT(user_request)
  specs = PARSE_SPECIFICATIONS(user_request)

  IF intent == "generate_code" THEN
    generated_code = GENERATE_CODE_FROM_SPECS(specs.description, specs.language, current_codebase_context)
    explanation = EXPLAIN_GENERATED_CODE(generated_code, specs.language)
    response = "Here's the code I generated:\n```" + specs.language + "\n" + generated_code + "\n```\n\nExplanation:\n" + explanation
  ELSEIF intent == "review_code" THEN
    code_to_review = specs.code_snippet OR GET_FILE_CONTENT(specs.file_path)
    linting_results = LINT_CODE(code_to_review, specs.language)
    vulnerability_report = CHECK_VULNERABILITIES(code_to_review, specs.language)
    style_suggestions = SUGGEST_STYLE_IMPROVEMENTS(code_to_review, specs.language)
    review_summary = COMBINE_REVIEW_RESULTS(linting_results, vulnerability_report, style_suggestions)
    response = "Code Review Summary:\n" + review_summary
  ELSEIF intent == "explain_code" THEN
    code_to_explain = specs.code_snippet OR GET_FILE_CONTENT(specs.file_path)
    explanation = EXPLAIN_EXISTING_CODE(code_to_explain, specs.language)
    response = "Explanation of the code:\n" + explanation
  ELSEIF intent == "refactor_code" THEN
    code_to_refactor = specs.code_snippet OR GET_FILE_CONTENT(specs.file_path)
    refactoring_goals = specs.refactor_goals // e.g., "improve performance", "increase readability"
    refactored_code = REFACTOR_CODE(code_to_refactor, specs.language, refactoring_goals)
    changes_explanation = EXPLAIN_REFACTORING_CHANGES(code_to_refactor, refactored_code)
    response = "Refactored code:\n```" + specs.language + "\n" + refactored_code + "\n```\n\nChanges made:\n" + changes_explanation
  ELSE
    response = "How can I assist you with your development task?"
  ENDIF

  RETURN response
END FUNCTION

FUNCTION GENERATE_CODE_FROM_SPECS(description, language, context)
  // Use a large language model (LLM) fine-tuned for code generation
  prompt = CREATE_CODE_GENERATION_PROMPT(description, language, context)
  generated_code = CALL_LLM_CODE_GENERATION(prompt)
  RETURN generated_code
END FUNCTION

FUNCTION LINT_CODE(code, language)
  // Use a language-specific linter tool (e.g., ESLint for JS, Pylint for Python)
  lint_output = EXECUTE_LINTER(code, language)
  RETURN PARSE_LINTER_OUTPUT(lint_output)
END FUNCTION

FUNCTION CHECK_VULNERABILITIES(code, language)
  // Use a static analysis security testing (SAST) tool
  sast_report = EXECUTE_SAST_TOOL(code, language)
  RETURN PARSE_SAST_REPORT(sast_report)
END FUNCTION

FUNCTION EXPLAIN_GENERATED_CODE(code, language)
  // Use an LLM to generate a natural language explanation of the code
  prompt = CREATE_CODE_EXPLANATION_PROMPT(code, language)
