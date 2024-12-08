import json
import os
import re
from typing import Any, Dict, List, Tuple

import weave
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
    LLMSynonymRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.gemini import Gemini
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import time

from graphrag_poc.custom_llama_index.custom_neo4j_property_graph import (
    Neo4jPropertyGraphStore,
)
from graphrag_poc.custom_llama_index.custom_vector_retriever import (
    CustomVectorContextRetriever,
)
from graphrag_poc.prompt_template import custom_prompt

load_dotenv()
weave.init("graphrag-poc")

openai_llm = AzureOpenAI(
    model=os.getenv("DEPLOYMENT_NAME"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0.0,
)
openai_embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    # deployment_name="my-custom-embedding",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
embeddings = openai_embed_model.get_text_embedding("Hello, world!")

gemini_llm = Gemini(
    model="models/gemini-1.5-flash-latest",
    temperature=0.0,
)
gemini_embed_model = GeminiEmbedding()
# embeddings = gemini_embed_model.get_text_embedding("Hello, world!")
# print(len(embeddings))


Settings.llm = gemini_llm
Settings.embed_model = gemini_embed_model


class MyResponse(BaseModel):
    question: str
    answer: str
    sources: List[NodeWithScore]


graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="12345678",
    url="bolt://localhost:7687",
    database="graphrag-openai-custom-extractor-prompt-gemini",
    refresh_schema=False,
)

prompt_template = (
    "You are a knowledge graph expert specializing in extracting structured information from text about startup founders, companies, and entrepreneurship guidance, particularly from Paul Graham's writings."
    "\nYour task is to extract up to {max_knowledge_triplets} knowledge triplets from the provided text. "
    "\n If there are more triplets that can be extracted, then extract as many as needed to capture all the information"
    "A knowledge triplet consists of (head, relation, tail) along with their types and properties."
    "\n\nCONTEXT AWARENESS:"
    "\n- Focus on startup-related entities: founders, companies, investors, concepts"
    "\n- Identify key entrepreneurship principles and advice"
    "\n- Capture relationships between people, organizations, and ideas"
    "\n- Extract time-sensitive information when available (founded_date, funding_rounds, etc.)"
    "\n---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Entity Properties: {allowed_entity_properties}\n"
    "Relation Types: {allowed_relation_types}\n"
    "Relation Properties: {allowed_relation_properties}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context.\n"
    "If the Entity Properties, Relation Properties contain property 'description', generate a context aware detailed description, which will have some uniqe non generic information addition"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'head_props': {{...}}, 'relation': '', 'relation_props': {{...}}, 'tail': '', 'tail_type': '', 'tail_props': {{...}}}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA') but where its ambiguous, use the entity as it is\n"
    "- Keep entities concise\n"
    "- While writing description property for entities and relations keep context in mind and just dont write the description of the entity or relation, but the description of the entity or relation in the context of the text\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "- While extracting relation, use singular form of the relation. Use EXPAND instead of EXPANDS or EXPECT instead of EXPECTS\n"
    "- The goal is to make relations as generics as possible, so that there are less duplicate relations in the graph, which have same meaning\n"
    "- If there are two names present in the text treat them as separate entities. For example Jessica Livingston and Robert Morris then they are two separate entities Jessica Livingston, Robert Morris\n"
    "- Focus on startup-specific metrics and relationships (funding rounds, valuations, mentor relationships)\n"
    "- Capture temporal aspects of relationships when mentioned (founding dates, acquisition dates)\n"
    "- Include relevant contextual properties (industry sector, technology stack, market focus)\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text Input: \nTim Cook, CEO of Apple Inc., announced the new Apple Watch that monitors heart health. "
    "UC Berkeley researchers studied the benefits of apples.\n"
    "Example Output:\n"
    "[{{'head': 'Tim Cook', 'head_type': 'PERSON', 'head_props': {{'description': 'Technology executive who made the product announcement for Apple Watch, demonstrating leadership in health-focused technology initiatives'}}, 'relation': 'CEO_OF', 'relation_props': {{'description': 'Executive leadership role involving product announcements and strategic health technology initiatives'}}, 'tail': 'Apple Inc.', 'tail_type': 'COMPANY', 'tail_props': {{'description': 'Technology company expanding into health monitoring through wearable devices'}}}},\n"
    " {{'head': 'Apple Inc.', 'head_type': 'COMPANY', 'head_props': {{'description': 'Company developing health-focused consumer technology products under Tim Cook's leadership'}}, 'relation': 'PRODUCE', 'relation_props': {{'description': 'Strategic initiative to enter health monitoring market through consumer devices'}}, 'tail': 'Apple Watch', 'tail_type': 'PRODUCT', 'tail_props': {{'description': 'Health-focused smartwatch representing Apple's expansion into medical monitoring technology'}}}},\n"
    " {{'head': 'Apple Watch', 'head_type': 'PRODUCT', 'head_props': {{'description': 'Wearable device specifically designed to track and monitor user health metrics'}}, 'relation': 'MONITOR', 'relation_props': {{'description': 'Continuous health monitoring capability focusing on cardiac metrics'}}, 'tail': 'heart health', 'tail_type': 'HEALTH_METRIC', 'tail_props': {{'description': 'Critical health metric monitored through Apple Watch's advanced sensors'}}}},\n"
    " {{'head': 'UC Berkeley', 'head_type': 'UNIVERSITY', 'head_props': {{'description': 'Academic institution conducting research on nutritional benefits and health impacts'}}, 'relation': 'STUDY', 'relation_props': {{'description': 'Academic research focusing on health benefits and nutritional value analysis'}}, 'tail': 'benefits of apples', 'tail_type': 'RESEARCH_TOPIC', 'tail_props': {{'description': 'Scientific investigation into the health advantages and nutritional properties of apples'}}}}]\n"
    "---------------------\n"
    "MAKE SURE TO FOLLOW THE EXAMPLE FORMAT STRICTLY FOR EACH KNOWLEDGE TRIPLET AND GIVE THE OUTPUT IN JSON LIST FORMAT ONLY AS ITS PARSED USING CODE AND IT NEEDS TO MATCH THE EXAMPLE FORMAT STRICTLY"
    "Text: {text}\n"
    "Output:\n"
)


def parse_dynamic_triplets_with_props(
    llm_output: str,
) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """
    Parse the LLM output and convert it into a list of entity-relation-entity triplets.
    This function is flexible and can handle various output formats.

    Args:
        llm_output (str): The output from the LLM, which may be JSON-like or plain text.

    Returns:
        List[Tuple[EntityNode, Relation, EntityNode]]: A list of triplets.
    """
    triplets = []

    try:
        # first extract the json from markdown
        llm_output = re.search(r"```json\s*([\s\S]*)\s*```", llm_output).group(1)
        # Attempt to parse the output as JSON
        data = json.loads(llm_output)
        for item in data:
            head = item.get("head")
            head_type = item.get("head_type")
            head_props = item.get("head_props", {})
            relation = item.get("relation")
            relation_props = item.get("relation_props", {})
            tail = item.get("tail")
            tail_type = item.get("tail_type")
            tail_props = item.get("tail_props", {})

            if head and head_type and relation and tail and tail_type:
                head_node = EntityNode(
                    name=head, label=head_type, properties=head_props
                )
                tail_node = EntityNode(
                    name=tail, label=tail_type, properties=tail_props
                )
                relation_node = Relation(
                    source_id=head_node.id,
                    target_id=tail_node.id,
                    label=relation,
                    properties=relation_props,
                )
                triplets.append((head_node, relation_node, tail_node))
    except json.JSONDecodeError:
        # Flexible pattern to match the key-value pairs for head, head_type, head_props, relation, relation_props, tail, tail_type, and tail_props
        pattern = r'[\{"\']head[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']head_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']head_props[\}"\']\s*:\s*\{(.*?)\}\s*,\s*[\{"\']relation[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']relation_props[\}"\']\s*:\s*\{(.*?)\}\s*,\s*[\{"\']tail[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']tail_type[\}"\']\s*:\s*[\{"\'](.*?)[\}"\']\s*,\s*[\{"\']tail_props[\}"\']\s*:\s*\{(.*?)\}\s*'

        # Find all matches in the output
        matches = re.findall(pattern, llm_output)

        for match in matches:
            (
                head,
                head_type,
                head_props,
                relation,
                relation_props,
                tail,
                tail_type,
                tail_props,
            ) = match

            # Use more robust parsing for properties
            def parse_props(props_str: str) -> Dict[str, Any]:
                try:
                    # Handle mixed quotes and convert to a proper dictionary
                    props_str = props_str.replace("'", '"')
                    return json.loads(f"{{{props_str}}}")
                except json.JSONDecodeError:
                    return {}

            head_props_dict = parse_props(head_props)
            relation_props_dict = parse_props(relation_props)
            tail_props_dict = parse_props(tail_props)

            head_node = EntityNode(
                name=head, label=head_type, properties=head_props_dict
            )
            tail_node = EntityNode(
                name=tail, label=tail_type, properties=tail_props_dict
            )
            relation_node = Relation(
                source_id=head_node.id,
                target_id=tail_node.id,
                label=relation,
                properties=relation_props_dict,
            )
            triplets.append((head_node, relation_node, tail_node))
    return triplets


dyn_llm_path_extractor = DynamicLLMPathExtractor(
    llm=Settings.llm,
    allowed_entity_props=["description"],
    allowed_relation_props=["description"],
    extract_prompt=prompt_template,
    parse_fn=parse_dynamic_triplets_with_props,
)


index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=Settings.llm,
    embed_model=Settings.embed_model,
    kg_extractors=[dyn_llm_path_extractor],
    show_progress=True,
)


llm_synonym_retriever = LLMSynonymRetriever(
    graph_store=graph_store,
    include_text=False,
    limit=100,
    max_keywords=20,
    llm=Settings.llm,
)

custom_vector_retriever = CustomVectorContextRetriever(
    graph_store,
    embed_model=Settings.embed_model,
    path_depth=2,
    include_text=True,
    similarity_score=0.7,
    limit=100,
    similarity_top_k=5,
    mode=VectorStoreQueryMode.HYBRID,
    hybrid_top_k=5,
)

query_engine = index.as_query_engine(
    sub_retrievers=[custom_vector_retriever, llm_synonym_retriever],
    text_qa_template=custom_prompt,
    response_mode="compact",
)

# query = 'What was the "class project" problem faced by young founders, and how can they overcome it?'
# response:RESPONSE_TYPE = query_engine.query(query)

# print("Response type: ", type(response))
# print(response)


# # create a my_response from response
# my_response = MyResponse(
#     question=query,
#     answer=response.response,
#     sources=response.source_nodes,
# )
# print(
#     my_response.model_dump_json(
#         indent=2
#     )
# )


def predict(query: str) -> Tuple[MyResponse, str]:
    response: RESPONSE_TYPE = query_engine.query(query)
    my_response = MyResponse(
        question=query,
        answer=response.response,
        sources=response.source_nodes,
    )
    return my_response, response.response

async def apredict(query: str) -> Tuple[MyResponse, str]:
    response: RESPONSE_TYPE = await query_engine.aquery(query)
    my_response = MyResponse(
        question=query,
        answer=response.response,
        sources=response.source_nodes,
    )
    return my_response, response.response

def process_questions_csv():
    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("artifacts")
    json_dumps_dir = artifacts_dir / "response_dumps_v3"
    json_dumps_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    csv_path = artifacts_dir / "questions_and_answers_v2.csv"
    df = pd.read_csv(csv_path)
    
    # Process each question that doesn't have an answer
    for index, row in df.iterrows():
        # if pd.isna(row['Answer']):
        if True:
            try:
                print(f"Processing question {index + 1}: {row['Question']}")
                
                # Get prediction
                my_response, answer = predict(row['Question'])
                
                # Update CSV
                # df.at[index, 'Answer'] = answer
                # df.to_csv(csv_path, index=False)
                
                json_path = json_dumps_dir / f"question_{index + 1}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(my_response.model_dump_json(indent=2))
                
                print(f"Saved answer and JSON dump for question {index + 1}")
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing question {index + 1}: {str(e)}")
                continue

if __name__ == "__main__":
    process_questions_csv()

