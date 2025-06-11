# Research & Summarization Agent

## Description
This agent performs research by scraping web pages and scholarly papers. It can synthesize information from multiple sources, include citations, and generate concise reports or executive summaries.

## Pseudocode
```
FUNCTION handle_research_request(topic, research_scope, output_format) // scope: "web", "scholarly", "all"
  // Source Identification and Scraping
  IF research_scope == "web" OR research_scope == "all" THEN
    web_sources = IDENTIFY_WEB_SOURCES(topic)
    scraped_web_content = SCRAPE_WEB_PAGES(web_sources)
  ELSE
    scraped_web_content = []
  ENDIF

  IF research_scope == "scholarly" OR research_scope == "all" THEN
    paper_databases = ["PubMed", "IEEE Xplore", "Google Scholar"] // Example databases
    scholarly_papers_metadata = SEARCH_SCHOLARLY_PAPERS(topic, paper_databases)
    // Potentially download PDFs if accessible
    scraped_papers_content = EXTRACT_TEXT_FROM_PAPERS(scholarly_papers_metadata)
  ELSE
    scraped_papers_content = []
  ENDIF

  all_scraped_content = COMBINE_CONTENT(scraped_web_content, scraped_papers_content)
  IF all_scraped_content IS EMPTY THEN
    RETURN "No relevant information found for '" + topic + "'."
  ENDIF

  // Information Extraction and Synthesis
  key_findings = EXTRACT_KEY_FINDINGS(all_scraped_content, topic)
  // Ensure proper citation tracking throughout synthesis
  synthesized_information = SYNTHESIZE_INFORMATION_WITH_CITATIONS(key_findings)

  // Report Generation
  IF output_format == "concise_report" THEN
    report = GENERATE_CONCISE_REPORT(synthesized_information, topic)
  ELSEIF output_format == "executive_summary" THEN
    summary = GENERATE_EXECUTIVE_SUMMARY(synthesized_information, topic)
    report = summary
  ELSEIF output_format == "bullet_points" THEN
    bullet_points = GENERATE_BULLET_POINT_SUMMARY(synthesized_information)
    report = FORMAT_AS_BULLET_LIST(bullet_points)
  ELSE
    report = "Invalid output format specified."
  ENDIF

  RETURN report
END FUNCTION

FUNCTION IDENTIFY_WEB_SOURCES(topic)
  // Use search engine APIs (Google, Bing) or targeted website searches
  search_query = topic + " reliable sources OR studies OR reports"
  search_results = CALL_SEARCH_ENGINE_API(search_query, count=20)
  filtered_urls = FILTER_RELEVANT_URLS(search_results)
  RETURN filtered_urls
END FUNCTION

FUNCTION SCRAPE_WEB_PAGES(urls)
  // Use libraries like BeautifulSoup, Scrapy
  content_list = []
  FOR EACH url IN urls
    html_content = FETCH_URL(url)
    text_content = EXTRACT_MAIN_TEXT_FROM_HTML(html_content)
    metadata = EXTRACT_METADATA(html_content) // author, date
    ADD_TO_LIST(content_list, CREATE_CONTENT_OBJECT(text_content, url, metadata.author, metadata.date))
  ENDFOR
  RETURN content_list
END FUNCTION

FUNCTION SEARCH_SCHOLARLY_PAPERS(topic, databases)
  // Use APIs for academic search engines (e.g., Semantic Scholar, CORE)
  papers_metadata = []
  FOR EACH db IN databases
    results = CALL_ACADEMIC_SEARCH_API(db, topic, count=10)
    ADD_ALL_TO_LIST(papers_metadata, results)
  ENDFOR
  // Deduplicate and rank papers
  unique_ranked_papers = PROCESS_SCHOLARLY_RESULTS(papers_metadata)
  RETURN unique_ranked_papers
END FUNCTION

FUNCTION EXTRACT_KEY_FINDINGS(content_list, research_topic)
  // Use NLP techniques: Named Entity Recognition, Relation Extraction, Topic Modeling
  findings = []
  FOR EACH content_item IN content_list
    entities = EXTRACT_NAMED_ENTITIES(content_item.text)
    relevant_sentences = FILTER_SENTENCES_BY_TOPIC(content_item.text, research_topic)
    summarized_points = SUMMARIZE_TEXT_CHUNKS(relevant_sentences)
    // Associate findings with their source for citation
    ADD_TO_LIST(findings, CREATE_FINDING_OBJECT(summarized_points, content_item.source_url, content_item.author, content_item.date))
  ENDFOR
  RETURN findings
END FUNCTION

FUNCTION SYNTHESIZE_INFORMATION_WITH_CITATIONS(key_findings_list)
  // Group related findings, identify themes, contradictions, and supporting evidence
  // Use LLM for coherent narrative generation, ensuring citations are maintained
  prompt = CREATE_SYNTHESIS_PROMPT(key_findings_list)
  synthesized_text = CALL_LLM_SYNTHESIS(prompt) // Model must be good at citation
  RETURN synthesized_text
END FUNCTION

FUNCTION GENERATE_CONCISE_REPORT(synthesized_text, topic)
  // Structure: Introduction, Key Sections based on themes, Conclusion, Bibliography
  report_structure = CREATE_REPORT_STRUCTURE(topic)
  POPULATE_REPORT_SECTIONS(report_structure, synthesized_text)
  FORMATTED_BIBLIOGRAPHY = GENERATE_BIBLIOGRAPHY_FROM_CITATIONS(synthesized_text)
  ADD_BIBLIOGRAPHY_TO_REPORT(report_structure, FORMATTED_BIBLIOGRAPHY)
  RETURN RENDER_REPORT(report_structure)
END FUNCTION
```
