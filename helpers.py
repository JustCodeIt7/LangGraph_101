import base64
from IPython.display import HTML, display
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod


def draw_mermaid_png(agent, max_height="70vh", max_width="min(90vw, 300px)"):

    png_bytes = agent.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )

    style = ";".join([
        "display:block",
        "margin:0 auto",
        "width:100%",
        f"max-width:{max_width}",
        f"max-height:{max_height}",
        "height:auto",
    ])

    encoded = base64.b64encode(png_bytes).decode("utf-8")
    html = f'<img src="data:image/png;base64,{encoded}" style="{style}"/>'
    display(HTML(html))