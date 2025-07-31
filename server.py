import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import tiktoken
from starlette.responses import JSONResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Graphrag and utility imports
from settings import load_settings_from_yaml
from utils import (
    load_parquet_files,
    convert_response_to_string,
    process_context_data,
    serialize_search_result,
)
from services.llm import setup_llm_and_embedder

from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities,
)

# Additional import for the OpenAI client
from openai import OpenAI

# --------------------------------------------------
# 1. Bootstrap environment & data loading
# --------------------------------------------------
load_dotenv()

# Initialize OpenAI client for cover letter editing
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

settings = load_settings_from_yaml("settings.yml")

# Initialize LLM & embedder
llm, text_embedder = setup_llm_and_embedder(settings)
# Token encoder for context builders
token_encoder = tiktoken.get_encoding("cl100k_base")

# Parquet & index loading
INPUT_DIR = settings.INPUT_DIR
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_LEVEL = settings.COMMUNITY_LEVEL
CLAIM_EXTRACTION_ENABLED = settings.GRAPHRAG_CLAIM_EXTRACTION_ENABLED
RESPONSE_TYPE = settings.RESPONSE_TYPE

(
    entity_df,
    entity_embedding_df,
    report_df,
    relationship_df,
    covariate_df,
    text_unit_df,
    community_df,
) = load_parquet_files(INPUT_DIR, CLAIM_EXTRACTION_ENABLED)

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
relationships = read_indexer_relationships(relationship_df)
claims = read_indexer_covariates(covariate_df) if CLAIM_EXTRACTION_ENABLED else []
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
text_units = read_indexer_text_units(text_unit_df)
communities = read_indexer_communities(community_df, entity_df, report_df)

# Vector stores
description_store = LanceDBVectorStore(collection_name="default-entity-description")
description_store.connect(db_uri=LANCEDB_URI)

full_store = LanceDBVectorStore(collection_name="default-community-full_content")
full_store.connect(db_uri=LANCEDB_URI)

# --------------------------------------------------
# 2. Setup search engines
# --------------------------------------------------

def setup_drift_search() -> DRIFTSearch:
    drift_reports = read_indexer_reports(
        report_df,
        entity_df,
        COMMUNITY_LEVEL,
        content_embedding_col="full_content_embeddings",
    )
    read_indexer_report_embeddings(drift_reports, full_store)
    drift_params = DRIFTSearchConfig(
        temperature=0,
        max_tokens=12_000,
        primer_folds=1,
        drift_k_followups=3,
        n_depth=3,
        n=1,
    )
    context_builder = DRIFTSearchContextBuilder(
        chat_llm=llm,
        text_embedder=text_embedder,
        entities=entities,
        relationships=relationships,
        reports=drift_reports,
        entity_text_embeddings=description_store,
        text_units=text_units,
        token_encoder=token_encoder,
        config=drift_params,
    )
    return DRIFTSearch(llm=llm, context_builder=context_builder, token_encoder=token_encoder)


def setup_global_search() -> GlobalSearch:
    try:
        enc = tiktoken.encoding_for_model(settings.GRAPHRAG_LLM_MODEL)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        token_encoder=enc,
    )
    return GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=enc,
        max_data_tokens=12_000,
        map_llm_params={"max_tokens": 1000, "temperature": 0.0, "response_format": {"type":"json_object"}},
        reduce_llm_params={"max_tokens": 2000, "temperature": 0.0},
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params={
            "use_community_summary": False,
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12_000,
            "context_name": "Reports",
        },
        concurrent_coroutines=32,
        response_type=RESPONSE_TYPE,
    )


def setup_local_search() -> LocalSearch:
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={"claims":claims} if CLAIM_EXTRACTION_ENABLED else None,
        entity_text_embeddings=description_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    return LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params={"max_tokens":2000, "temperature":0.0},
        context_builder_params={
            "text_unit_prop":0.5,
            "community_prop":0.1,
            "conversation_history_max_turns":5,
            "conversation_history_user_turns_only":True,
            "top_k_mapped_entities":10,
            "top_k_relationships":10,
            "include_entity_rank":True,
            "include_relationship_weight":True,
            "include_community_rank":False,
            "return_candidate_context":False,
            "embedding_vectorstore_key":EntityVectorStoreKey.ID,
            "max_tokens":12_000,
        },
        response_type=RESPONSE_TYPE,
    )

global_search_engine = setup_global_search()
local_search_engine = setup_local_search()
drift_search_engine = setup_drift_search()

# --------------------------------------------------
# 3. Define Pydantic models for search outputs
# --------------------------------------------------
class SearchResponse(BaseModel):
    response: str
    context_data: List[Dict]
    context_text: str
    completion_time: float
    llm_calls: int
    llm_calls_categories: Dict[str,int]
    output_tokens: int
    output_tokens_categories: Dict[str,int]
    prompt_tokens: int
    prompt_tokens_categories: Dict[str,int]

class GlobalSearchResponse(SearchResponse):
    reduce_context_data: List[Dict]
    reduce_context_text: str
    map_responses: List[Dict]

# --------------------------------------------------
# 4. Initialize FastAPI + MCP
# --------------------------------------------------

mcp = FastMCP(name="GraphragMCP")
app = FastAPI()

# CORS for front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Expose a status endpoint returning 200
@app.get("/status")
async def status():
    return {"status": "Server is up and running"}

# --------------------------------------------------
# 5. Register graph-search tools
# --------------------------------------------------
@mcp.tool(description="Perform a DRIFT search on the graph data")
async def drift_search(query: str) -> SearchResponse:
    result = await drift_search_engine.asearch(query)
    return SearchResponse(
        response=convert_response_to_string(result.response.nodes[0].answer),
        context_data=process_context_data(result.context_data),
        context_text=result.context_text,
        completion_time=result.completion_time,
        llm_calls=result.llm_calls,
        llm_calls_categories=result.llm_calls_categories,
        output_tokens=result.output_tokens,
        output_tokens_categories=result.output_tokens_categories,
        prompt_tokens=result.prompt_tokens,
        prompt_tokens_categories=result.prompt_tokens_categories,
    )

@mcp.tool(description="Perform a global search on the graph data")
async def global_search(query: str) -> GlobalSearchResponse:
    result = await global_search_engine.asearch(query)
    return GlobalSearchResponse(
        response=convert_response_to_string(result.response),
        context_data=process_context_data(result.context_data),
        context_text=result.context_text,
        reduce_context_data=process_context_data(result.reduce_context_data),
        reduce_context_text=result.reduce_context_text,
        map_responses=[serialize_search_result(r) for r in result.map_responses],
        completion_time=result.completion_time,
        llm_calls=result.llm_calls,
        llm_calls_categories=result.llm_calls_categories,
        output_tokens=result.output_tokens,
        output_tokens_categories=result.output_tokens_categories,
        prompt_tokens=result.prompt_tokens,
        prompt_tokens_categories=result.prompt_tokens_categories,
    )


@mcp.tool(description="Perform a local search on the graph data")
async def local_search(query: str) -> SearchResponse:
    print(f"[MCP] local_search called with query: {query!r}")
    result = await local_search_engine.asearch(query)
    return SearchResponse(
        response=convert_response_to_string(result.response),
        # Wrap context_data in a list to satisfy the SearchResponse List[Dict] type
        context_data=[process_context_data(result.context_data)],
        context_text=result.context_text,
        completion_time=result.completion_time,
        llm_calls=result.llm_calls,
        llm_calls_categories=result.llm_calls_categories,
        output_tokens=result.output_tokens,
        output_tokens_categories=result.output_tokens_categories,
        prompt_tokens=result.prompt_tokens,
        prompt_tokens_categories=result.prompt_tokens_categories,
    )

# Direct FastAPI route to proxy POST requests to /mcp/local_search
from fastapi import Request

@app.post("/mcp/local_search")
async def proxy_local_search(request: Request):
    payload = await request.json()
    query = payload.get("query")
    # Call the same local_search tool logic
    return await local_search(query)

# --------------------------------------------------
# 6. Placeholders for cover-letter agents
# --------------------------------------------------
@mcp.tool(description="Fetch company overview, values, and recent news")
async def company_info(name: str) -> Dict:
    # TODO: implement
    return {}

@mcp.tool(description="Lookup hiring manager for a given company and role")
async def hiring_manager(company: str, role: str) -> Dict:
    # TODO: implement
    return {}

@mcp.tool(description="Draft an initial cover letter from research inputs")
async def draft_cover_letter(
    company_info: Dict,
    hiring_manager: Dict,
    projects: List[Dict],
    job_description: str
) -> Dict[str,str]:
    # TODO: implement
    return {"draft": ""}

@mcp.tool(description="Evaluate a draft cover letter for relevance and style")
async def evaluate_draft(draft: str, job_description: str) -> Dict[str,object]:
    # TODO: implement
    return {"score":0.0, "suggestions":[]}

@mcp.tool(description="Apply suggestions to finalize the cover letter")
async def finalize_cover_letter(draft: str, suggestions: List[str]) -> Dict[str,str]:
    # TODO: implement
    return {"final_letter": ""}

# New MCP tool for editing a cover letter via OpenAI Chat API
@mcp.tool(description="Edit a cover letter via chat using OpenAI API")
async def edit_cover_letter(conversation: List[str], current_letter: str) -> Dict[str, str]:
    # Construct the prompt for the cover letter editor
    prompt = (
        "You are an expert cover letter editor. "
        "Below is the current cover letter followed by a conversation history of requested edits. "
        "Please generate an updated cover letter that incorporates all the feedback.\n\n"
        "Current cover letter:\n"
        f"{current_letter}\n\n"
        "Conversation history:\n"
        f"{ '\n'.join(conversation) }\n\n"
        "Updated cover letter:"
    )
    # For debugging: log the prompt
    print("Generated prompt:", prompt)
    # Use OpenAI ChatCompletion API to edit the cover letter
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert cover letter editor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=2000
    )
    updated_letter = response.choices[0].message.content
    # Debug log the updated letter
    print("Updated letter:", updated_letter)
    return {"updatedLetter": updated_letter}

# Proxy route for the edit cover letter tool
@app.post("/mcp/edit_cover_letter")
async def proxy_edit_cover_letter(request: Request):
    payload = await request.json()
    conversation = payload.get("conversation", [])
    current_letter = payload.get("currentLetter", "")
    # Call the edit_cover_letter tool and return its result
    return await edit_cover_letter(conversation=conversation, current_letter=current_letter)

# Mount the Streamable HTTP transport under /mcp with CORS
mcp_app = mcp.streamable_http_app()
# Enable CORS on the MCP sub-app so browser clients can access it
mcp_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/mcp", mcp_app)

# --------------------------------------------------
# 7. Run the server
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)