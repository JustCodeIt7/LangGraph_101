---
mode: "agent"
model: "gemini-2.5-pro"
tools: ["githubRepo", "codebase"]
description: "Generate a new React form component"
---

Your goal is to transform a raw Python script into a well-commented, educational resource suitable for a programming tutorial. Add concise, helpful comments and structured section headers to make the code easy for viewers to understand and follow.

## Guidelines

**General**

1. **Return Only Code:** Your final output should only be the fully commented Python code. Do not include any extra explanations, summaries, or introductions.

2. **No Import Comments:** Leave import statements uncommented.

**Section Banners**

3. **Use Clear Headers:** Divide the script into logical sections using banner comments. The banner format must be _exact_:

4. ################################ Section Title ################################

**Commenting Style**

4. **Be Concise & Purpose-Driven:** Write short, useful comments that explain the _why_, not the _what_. Avoid obvious statements.

▪ **Bad:** i = i + 1  # Increment i

▪ **Good:** retries += 1 # Increment retry counter for the next attempt

5. **Use the Imperative Mood:** Start comments with a command verb.

▪ **Instead of:** This function will validate the data.

▪ **Use:** Validate the input data.

6. **Block Comments:** For multi-line logic (like a try/except block, a with statement, or a complex loop), place a single comment on the line directly **above** the block, explaining its overall purpose.

7. **Inline Comments:** Place inline comments at the end of a line to clarify non-obvious function arguments, complex calculations, or tricky implementation details.

8. **Conditional Logic:** Add a comment before if/elif/else blocks or for/while loops _only if_ the condition or the purpose of the loop isn't immediately obvious.

## Enhanced Reference Example

This example demonstrates the target style, including section banners, block comments, a for loop, and inline comments for clarifying specific arguments.

```python
################################ App Configuration & Setup ################################

# Configure the main application settings
app.config.update(
    SECRET_KEY=os.environ.get("APP_SECRET"),    # Load secret key from environment variable
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",              # Mitigate CSRF attacks
)

# Establish a connection to the database
with get_db_connection() as conn:
    # Check if the 'users' table is missing and create it if necessary
    if not table_exists("users", conn):
        create_user_table(conn) # Initialize the default user schema

################################ API Endpoints ################################

@app.route("/api/v1/users", methods=["GET"])
def get_all_users():
    # Retrieve sorting preference from query parameters, defaulting to 'id'
    sort_key = request.args.get("sort_by", "id")

    users = fetch_users_from_db(sort_by=sort_key)

    # Prepare user data for the response
    formatted_users = []
    # Loop through each database record to build the public user object
    for user in users:
        formatted_users.append({
            "id": user.id,
            "username": user.username,
            "profile_url": f"/users/{user.id}" # Generate a dynamic URL for the user
        })

    return jsonify(formatted_users)

```

## Code to Comment
