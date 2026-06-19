import logging
import json
import ollama
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END

from backend.agent.retriever import retrieve_hierarchical_context
from backend.agent.tools import get_pdf_metadata, get_tables
from backend.agent.reflection import evaluate_evidence_sufficiency, generate_expanded_query

logger = logging.getLogger(__name__)

# State definition
class AgentState(TypedDict):
    pdf_id: int
    query: str
    chat_history: List[Dict[str, Any]]
    
    # Internal variables
    query_type: str             # 'summary', 'comparison', 'factual', 'table_lookup', 'metadata'
    complexity: str             # 'simple' or 'complex'
    model_name: str             # active Ollama model name
    current_search_query: str
    search_attempts: List[str]
    retrieved_context: str
    raw_evidence_chunks: List[Dict[str, Any]]
    thought_process: List[str]  # Appended during execution
    evidence_sufficient: bool
    response: str               # Final synthesized response text

# Node 1: Classify Query & Complexity
def classify_query_node(state: AgentState, model_name: str = "qwen2.5:7b") -> Dict[str, Any]:
    query = state["query"]
    logger.info(f"Node: Classify Query for '{query}' using model {model_name}")
    
    prompt = (
        "You are an intelligent query router for a PDF document intelligence system.\n"
        "Analyze the user query and classify it.\n\n"
        "1. Classification Types:\n"
        "   - 'summary': General summaries of the document, chapters, or large sections (e.g. 'Summarize this PDF', 'What is this document about?').\n"
        "   - 'table_lookup': Looking for specific values, numbers, or tables (e.g. 'Show me the table on page 4', 'What is the revenue table?').\n"
        "   - 'metadata': Inquiring about document properties (e.g. 'What is the filename?', 'How many pages?', 'Who is the author?').\n"
        "   - 'factual': Direct questions about specific sections or facts (e.g. 'What is the compliance policy?', 'What does clause 4.2 say?').\n"
        "   - 'comparison': Asking to compare, contrast, or find contradictions (e.g. 'Compare 2024 vs 2025 revenue', 'Find contradictions between safety and compliance').\n\n"
        "2. Complexity:\n"
        "   - 'simple': Basic direct lookups, metadata requests, or straightforward factual questions.\n"
        "   - 'complex': Queries requiring synthesis across multiple sections, comparisons, trends, or multi-step reasoning.\n\n"
        "Respond in exactly this format:\n"
        "TYPE: <type>\n"
        "COMPLEXITY: <simple/complex>\n\n"
        f"User Query: {query}"
    )
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        content = response["message"]["content"].strip()
        logger.info(f"Raw LLM classification response:\n{content}")
        lines = content.split("\n")
        
        query_type = "factual"
        complexity = "simple"
        
        import re
        for line in lines:
            if "type" in line.lower():
                val = line.split(":")[-1].strip().lower()
                val = "".join([c for c in val if c.isalnum()])
                if val in ["summary", "table_lookup", "metadata", "factual", "comparison"]:
                    query_type = val
            if "complexity" in line.lower():
                val = line.split(":")[-1].strip().lower()
                val = "".join([c for c in val if c.isalnum()])
                if val in ["simple", "complex"]:
                    complexity = val
                    
        # Force metadata/table_lookup to be simple to bypass expensive operations
        if query_type in ["metadata", "table_lookup"]:
            complexity = "simple"
            
        thought = f"Classified query: Type = '{query_type}', Complexity = '{complexity}'."
        logger.info(thought)
        return {
            "query_type": query_type,
            "complexity": complexity,
            "thought_process": [thought]
        }
    except Exception as e:
        logger.error(f"Error in classify_query_node: {str(e)}")
        return {
            "query_type": "factual",
            "complexity": "simple",
            "thought_process": ["Failed query classification; falling back to 'factual' and 'simple'."]
        }

# Node 2: Retrieve Evidence
def retrieve_evidence_node(state: AgentState) -> Dict[str, Any]:
    pdf_id = state["pdf_id"]
    query = state["query"]
    current_search = state["current_search_query"]
    query_type = state["query_type"]
    complexity = state.get("complexity", "simple")
    model_name = state.get("model_name", "qwen2.5:7b")
    
    logger.info(f"Node: Retrieve Evidence for PDF {pdf_id} with query '{current_search}' (Type: {query_type}, Complexity: {complexity})")
    
    thoughts = []
    retrieved_context = ""
    raw_evidence = []
    
    if query_type == "metadata":
        thoughts.append("Retrieving document metadata from database...")
        retrieved_context = get_pdf_metadata(pdf_id)
        
    elif query_type == "table_lookup":
        thoughts.append("Searching for structural tables extracted by Docling...")
        retrieved_context = get_tables(pdf_id)
        
    elif query_type == "summary":
        thoughts.append("Gathering parent section summaries for document summarization...")
        # Fetch parent section summaries from Qdrant directly
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from backend.agent.retriever import qdrant_client
        
        hits, _ = qdrant_client.scroll(
            collection_name="parent_sections",
            scroll_filter=Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
            ),
            limit=50
        )
        if hits:
            summaries = []
            for hit in hits:
                summaries.append(f"Section: {hit.payload['heading']}\nSummary: {hit.payload['summary']}")
            retrieved_context = "\n\n".join(summaries)
        else:
            retrieved_context = "No structural sections found to summarize."
            
    else: # factual or comparison
        # Adaptive Retrieval: retrieve fewer chunks for simple queries
        top_chunks = 3 if complexity == "simple" else 5
        
        thoughts.append(f"Searching relevant sections (Adaptive Retrieval) for '{current_search}'...")
        chunks = retrieve_hierarchical_context(
            pdf_id, 
            current_search, 
            top_sections=3, 
            top_chunks=top_chunks, 
            complexity=complexity, 
            model_name=model_name
        )
        raw_evidence = chunks
        
        if chunks:
            formatted_results = []
            for idx, chunk in enumerate(chunks):
                formatted_results.append(
                    f"Source: Page {chunk['page_number']}, Section: {chunk['heading']}\n"
                    f"Content: {chunk['content']}"
                )
            retrieved_context = "\n\n".join(formatted_results)
            thoughts.append(f"Retrieved {len(chunks)} relevant evidence chunks from matching sections.")
        else:
            retrieved_context = "No relevant document sections found."
            thoughts.append("Warning: Could not find any relevant text matching the query.")
            
    return {
        "retrieved_context": retrieved_context,
        "raw_evidence_chunks": raw_evidence,
        "thought_process": thoughts
    }

# Node 3: Evaluate Evidence (Reflection)
def evaluate_evidence_node(state: AgentState, model_name: str = "qwen2.5:7b") -> Dict[str, Any]:
    query = state["query"]
    context = state["retrieved_context"]
    query_type = state["query_type"]
    complexity = state.get("complexity", "simple")
    raw_evidence = state.get("raw_evidence_chunks", [])
    search_attempts = state["search_attempts"]
    
    logger.info("Node: Evaluate Evidence (Reflection)")
    
    # Non-search queries don't need reflection
    if query_type in ["metadata", "table_lookup", "summary"]:
        return {
            "evidence_sufficient": True,
            "thought_process": ["Sufficient context gathered for synthesis."]
        }
        
    # Conditional Reflection Check
    if not raw_evidence:
        # No chunks found at all: force query expansion
        thoughts = ["No matching evidence found. Triggering query expansion..."]
        if len(search_attempts) < 2:
            new_query = generate_expanded_query(query, search_attempts, model_name=model_name)
            thoughts.append(f"Query expanded to '{new_query}'. Retrying search...")
            return {
                "evidence_sufficient": False,
                "current_search_query": new_query,
                "search_attempts": search_attempts + [new_query],
                "thought_process": thoughts
            }
        else:
            thoughts.append("Search retry limit reached. Proceeding with empty response.")
            return {
                "evidence_sufficient": True,
                "thought_process": thoughts
            }
            
    if complexity == "simple":
        # Skip LLM reflection check for simple queries
        logger.info("Adaptive Agent: Skipping LLM reflection check for Simple query.")
        return {
            "evidence_sufficient": True,
            "thought_process": ["Simple query: skipping LLM reflection check."]
        }
        
    # Run full LLM-based reflection for complex queries
    thoughts = ["Analyzing evidence sufficiency against user request..."]
    sufficient = evaluate_evidence_sufficiency(query, context, model_name=model_name)
    
    if sufficient:
        thoughts.append("Evidence is sufficient. Moving to response generation.")
        return {
            "evidence_sufficient": True,
            "thought_process": thoughts
        }
        
    # Insufficient evidence: attempt query expansion if we haven't reached limits
    if len(search_attempts) < 2:
        thoughts.append("Evidence is insufficient. Triggering query expansion...")
        new_query = generate_expanded_query(query, search_attempts, model_name=model_name)
        thoughts.append(f"Query expanded to '{new_query}'. Retrying search...")
        
        return {
            "evidence_sufficient": False,
            "current_search_query": new_query,
            "search_attempts": search_attempts + [new_query],
            "thought_process": thoughts
        }
    else:
        thoughts.append("Evidence is insufficient, but search retry limit reached. Proceeding with available details.")
        return {
            "evidence_sufficient": True,
            "thought_process": thoughts
        }

# Define the LangGraph StateGraph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("classify", classify_query_node)
workflow.add_node("retrieve", retrieve_evidence_node)
workflow.add_node("evaluate", evaluate_evidence_node)

# Add Edges
workflow.set_entry_point("classify")
workflow.add_edge("classify", "retrieve")
workflow.add_edge("retrieve", "evaluate")

# Conditional Routing
def route_reflection(state: AgentState) -> Literal["retrieve", END]:
    if state["evidence_sufficient"]:
        return END
    return "retrieve"

workflow.add_conditional_edges(
    "evaluate",
    route_reflection,
    {
        "retrieve": "retrieve",
        END: END
    }
)

# Compile the Graph
agent_graph = workflow.compile()

# Execution stream helper
def run_agent_stream(pdf_id: int, query: str, chat_history: List[Dict[str, Any]], model_name: str = "qwen2.5:7b"):
    """
    Runs the LangGraph agent stream on a document using the selected model.
    """
    state = {
        "pdf_id": pdf_id,
        "query": query,
        "chat_history": chat_history,
        "query_type": "factual",
        "complexity": "simple",
        "model_name": model_name,
        "current_search_query": query,
        "search_attempts": [query],
        "retrieved_context": "",
        "raw_evidence_chunks": [],
        "thought_process": [],
        "evidence_sufficient": False,
        "response": ""
    }
    
    logger.info(f"Executing Agent LangGraph state machine with model '{model_name}'...")
    
    # 1. Run Classification Node first
    update = classify_query_node(state, model_name=model_name)
    state.update(update)
    for thought in state["thought_process"]:
        yield f"data: {json.dumps({'type': 'thought', 'content': thought})}\n\n"
    state["thought_process"] = []
    
    step_limit = 10
    step = 0
    
    # 2. Bypass state machine loop and reflection checks for simple, metadata, or table queries
    if state["complexity"] == "simple" or state["query_type"] in ["metadata", "table_lookup"]:
        logger.info("Bypassing LangGraph loop and reflection node for simple/metadata/table query")
        update = retrieve_evidence_node(state)
        state.update(update)
        for thought in state["thought_process"]:
            yield f"data: {json.dumps({'type': 'thought', 'content': thought})}\n\n"
        state["thought_process"] = []
    else:
        # Complex queries run retrieve -> evaluate (reflection) -> loop
        current_node = "retrieve"
        while current_node != END and step < step_limit:
            step += 1
            logger.info(f"LangGraph step {step}: node '{current_node}'")
            
            if current_node == "retrieve":
                update = retrieve_evidence_node(state)
                state.update(update)
                current_node = "evaluate"
                
            elif current_node == "evaluate":
                update = evaluate_evidence_node(state, model_name=model_name)
                state.update(update)
                current_node = route_reflection(state)
                
            for thought in state["thought_process"]:
                yield f"data: {json.dumps({'type': 'thought', 'content': thought})}\n\n"
            state["thought_process"] = []
        
    yield f"data: {json.dumps({'type': 'thought', 'content': 'Synthesizing grounded response...'})}\n\n"
    
    system_prompt = (
        "You are an advanced, helpful document intelligence assistant. "
        "Your goal is to answer the user query based ONLY on the provided context from the PDF document. "
        "Adhere to the following rules strictly:\n"
        "1. Answer the question factually and directly using the context.\n"
        "2. For every fact or claim you make, cite the page number from the context using the exact format '[Page N]' (e.g. '[Page 4]'). "
        "Do not invent citation pages.\n"
        "3. Quote short exact sentences or key terms from the context where appropriate.\n"
        "4. If the context does not contain the answer, state clearly: 'I am sorry, but the provided document does not contain the information necessary to answer this question.'\n"
        "5. Do not make up facts or extrapolate beyond the text.\n"
        "6. PDF text extraction can sometimes be jumbled or out of order (e.g., candidate/student names mixed into sentence structures). "
        "Be careful not to confuse names of individuals (e.g., candidates/students) with names of institutions, colleges, or programs. "
        "Look for other clues in the document like web domains (e.g. www.vit.ac.in), emails, or footer text to correctly identify organizations."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    user_prompt = (
        f"Retrieved Document Context:\n"
        f"============================\n"
        f"{state['retrieved_context']}\n"
        f"============================\n\n"
        f"User Query: {state['query']}"
    )
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response_stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options={"temperature": 0.3}
        )
        
        for chunk in response_stream:
            token = chunk["message"]["content"]
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
    except Exception as e:
        logger.error(f"Error in streaming generation: {str(e)}")
        yield f"data: {json.dumps({'type': 'token', 'content': f'Error generating response: {str(e)}'})}\n\n"
        
    sources = []
    for chunk in state["raw_evidence_chunks"]:
        sources.append({
            "content": chunk["content"],
            "page_number": chunk["page_number"],
            "heading": chunk["heading"]
        })
        
    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
    yield "event: end\ndata: {}\n\n"
