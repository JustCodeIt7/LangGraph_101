## Human-in-the-loop Advanced

### Graph Structure

``` mermaid
graph TD
    START([START]) --> improve_prompt[improve_prompt]
    improve_prompt --> human_review[human_review]
    human_review --> answer_prompt[answer_prompt]
    answer_prompt --> END([END])
    
    class START,END startEnd
    class improve_prompt,human_review,answer_prompt processNode

```

``` mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__(<p>__start__</p>)
	get_approval(get_approval)
	router(router)
	complete_task(complete_task)
	cancel_task(cancel_task)
	__end__(<p>__end__</p>)
	__start__ --> get_approval;
	get_approval --> router;
	router --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```