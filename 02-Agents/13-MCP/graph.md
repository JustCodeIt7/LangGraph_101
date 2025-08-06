# Graph Representation of Agent State

```  mermaid
flowchart TD
    %% --- State Definition ---
    subgraph AgentState
        direction TB
        M[Messages<br/>(List[BaseMessage])]
    end

    %% --- Graph Nodes ---
    START(Entry&nbsp;Point)
    A[agent<br/>(LLM&nbsp;decides&nbsp;next&nbsp;action)]
    T[tools<br/>(execute&nbsp;requested&nbsp;tool)]
    END((END))

    %% --- Edges ---
    START --> A
    A --|"continue"|--> T
    A --|"end"|--> END
    T --> A

    %% --- Styling ---
    classDef node fill:#f8f8ff,stroke:#333,stroke-width:1px;
    class START,END node;
    class A,T node;

```
