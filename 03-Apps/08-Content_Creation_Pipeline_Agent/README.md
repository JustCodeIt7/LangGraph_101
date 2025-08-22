# Content Creation Pipeline Agent

## Description
This agent automates parts of the content creation process. It can research topics, outline blog posts or video scripts, provide SEO suggestions, adapt content for social media, and curate automated newsletters.

## Pseudocode
```
FUNCTION handle_content_creation_request(request_type, topic, target_audience, existing_content)
  IF request_type == "generate_blog_outline" THEN
    research_data = RESEARCH_TOPIC(topic)
    competitor_analysis = ANALYZE_COMPETITOR_CONTENT(topic)
    outline = GENERATE_BLOG_POST_OUTLINE(topic, research_data, competitor_analysis, target_audience)
    seo_keywords = SUGGEST_SEO_KEYWORDS(topic, research_data)
    response = "Blog Post Outline for '" + topic + "':\n" + FORMAT_OUTLINE(outline) + "\n\nSuggested SEO Keywords: " + JOIN_STRINGS(seo_keywords, ", ")
  ELSEIF request_type == "generate_video_script_outline" THEN
    research_data = RESEARCH_TOPIC(topic)
    visual_ideas = BRAINSTORM_VISUAL_IDEAS(topic)
    script_outline = GENERATE_VIDEO_SCRIPT_OUTLINE(topic, research_data, visual_ideas, target_audience)
    response = "Video Script Outline for '" + topic + "':\n" + FORMAT_SCRIPT_OUTLINE(script_outline)
  ELSEIF request_type == "adapt_for_social_media" THEN
    IF existing_content IS NULL THEN
      response = "Please provide the existing content to adapt."
    ELSE
      platform = DETECT_SOCIAL_MEDIA_PLATFORM(topic) // Or specified by user
      adapted_posts = ADAPT_CONTENT_FOR_PLATFORM(existing_content, platform, target_audience)
      response = "Adapted Social Media Posts for " + platform + ":\n" + FORMAT_SOCIAL_POSTS(adapted_posts)
    ENDIF
  ELSEIF request_type == "curate_newsletter" THEN
    content_sources = GET_NEWSLETTER_CONTENT_SOURCES(topic)
    recent_articles = FETCH_RECENT_ARTICLES(content_sources)
    curated_items = SELECT_AND_SUMMARIZE_ARTICLES(recent_articles, target_audience, count=5)
    newsletter_draft = COMPOSE_NEWSLETTER(curated_items, topic)
    response = "Draft Newsletter for '" + topic + "':\n" + newsletter_draft
  ELSE
    response = "What content creation task can I help you with?"
  ENDIF

  RETURN response
END FUNCTION

FUNCTION RESEARCH_TOPIC(topic)
  // Use web search, knowledge bases, APIs
  search_results = WEB_SEARCH(topic + " trends", topic + " statistics", topic + " expert opinions")
  key_information = EXTRACT_KEY_INFORMATION(search_results)
  RETURN key_information
END FUNCTION

FUNCTION GENERATE_BLOG_POST_OUTLINE(topic, research, competitor_analysis, audience)
  // Use LLM to structure content logically
  prompt = CREATE_BLOG_OUTLINE_PROMPT(topic, research, competitor_analysis, audience)
  outline_structure = CALL_LLM_CONTENT_GENERATION(prompt)
  RETURN outline_structure
END FUNCTION

FUNCTION SUGGEST_SEO_KEYWORDS(topic, research_data)
  // Use SEO tools API or LLM with SEO knowledge
  potential_keywords = EXTRACT_KEYWORDS_FROM_TEXT(research_data)
  // Add keyword research tool integration here
  relevant_keywords = FILTER_AND_RANK_KEYWORDS(potential_keywords, topic)
  RETURN relevant_keywords
END FUNCTION

FUNCTION ADAPT_CONTENT_FOR_PLATFORM(content, platform, audience)
  // Tailor tone, length, hashtags, visuals for specific platforms (Twitter, LinkedIn, Instagram etc.)
  adaptation_rules = GET_PLATFORM_ADAPTATION_RULES(platform)
  adapted_versions = []
  FOR EACH rule_set IN adaptation_rules
    adapted_post = APPLY_ADAPTATION(content, rule_set, audience)
    ADD_TO_LIST(adapted_versions, adapted_post)
  ENDFOR
  RETURN adapted_versions
END FUNCTION

FUNCTION COMPOSE_NEWSLETTER(curated_items, topic_theme)
  // Use a template and fill with summarized items, intro, outro
  newsletter_html = START_NEWSLETTER_TEMPLATE(topic_theme)
  ADD_INTRO_SECTION(newsletter_html, topic_theme)
  FOR EACH item IN curated_items
    ADD_CURATED_ITEM_TO_NEWSLETTER(newsletter_html, item.title, item.summary, item.source_url)
  ENDFOR
  ADD_OUTRO_SECTION(newsletter_html, topic_theme)
  RETURN newsletter_html
END FUNCTION
```

## Installation & Usage

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) installed and running

### Quick Start

1. **Setup the environment:**
   ```bash
   python setup.py
   ```

2. **Run the agent:**
   ```bash
   python app.py
   ```

### Manual Setup (Alternative)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama and pull llama3.2:**
   ```bash
   # In a separate terminal
   ollama serve
   
   # Pull the model
   ollama pull llama3.2
   ```

3. **Run the agent:**
   ```bash
   python app.py
   ```

## Usage Examples

The agent can handle various content creation requests:

- **Blog Outline:** "Create a blog outline about sustainable energy"
- **Video Script:** "Generate a video script about productivity tips"
- **Social Media:** "Adapt this content for social media platforms"
- **Newsletter:** "Curate a newsletter about AI trends"

## Features

✅ **Blog Post Outlines** - Structured outlines with SEO keywords
✅ **Video Script Outlines** - Engaging scripts with visual suggestions
✅ **Social Media Adaptation** - Platform-specific content for Twitter, LinkedIn, Instagram
✅ **Newsletter Curation** - Professional newsletter drafts
✅ **Topic Research** - Automated research and insights
✅ **LangGraph Workflow** - Structured agent pipeline
✅ **Local LLM** - Uses Ollama with llama3.2 for privacy

## Architecture

The agent uses LangGraph to create a stateful workflow:
1. **Parse Request** - Understand user intent
2. **Research Topic** - Gather relevant information
3. **Route & Generate** - Create content based on request type
4. **Output** - Return formatted content

Built with:
- **LangGraph** - Agent orchestration framework
- **Ollama** - Local LLM inference
- **llama3.2** - Language model
- **LangChain** - LLM integration utilities
