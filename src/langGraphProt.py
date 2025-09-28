print("Import initialized.")
import time
startImp = time.perf_counter()
import os
import requests
print("Loading FAISS and embedding model, please wait...")
import faiss
from dataLoader import embeddingModel, chunks
import numpy as np
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import HumanMessage, AIMessage
from sklearn.metrics.pairwise import cosine_similarity
import re
endImp = time.perf_counter()
print(f"Import successful (took {endImp-startImp:.1f} seconds). Initializing graph...")
#At this point, all imports are done (timed for performance)



##ADJUSTABLE PARAMETERS##
os.environ["API_KEY"] = "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"
#Replace with your API key
SITE_URL = "https://openrouter.ai/api/v1/chat/completions"
#Replace with a preferred site URL
langModel = "meta-llama/llama-3.3-70b-instruct:free"
#Replace with a preferred model
kValue = 5
#Replace with a preferred amount of chunks (context) to fetch
minCoverage = 0.7
#Replace with a preferred minimum of keyword coverage in the retrieved chunks
maxDistance = 1.2
#Replace with a preferred max distance of similarity



faissIndexPath = "faiss_index/chunk_index.faiss"
#Contains FAISS index, used for quick and efficient similarity search - created upon running dataLoader.py
index = faiss.read_index(faissIndexPath)
with open("faiss_index/chunks.json", "r", encoding="utf-8") as f:
    chunk_texts = json.load(f)
    #Read the chunks as text too to pass on to the LLM alongside with the query
print(f"Chunks successfully loaded.")


class State(TypedDict):
    messages: Annotated[list, add_messages] 
    retrieval: dict
graph_builder = StateGraph(State)


#A simple function to retrieve keywords from a query
def getKeywords(text):
    commonWords = {"a", "an", "the", "is", "and", "or", "for", "to", "with", "of", "on", "in"}
    #Words to ignore
    allWords = re.findall(r"\w+", text.lower())
    keywords = [w for w in allWords if w not in commonWords]
    #Retrieve keywords
    return keywords
#A simple function that returns the percentage of keywords covered in the retrieved chunks
def keywordCoverage(query, chunks):
    keywords = set(getKeywords(query))
    #Retrieve keywords using the above function
    chunk_text = " ".join(chunks).lower()
    matched = [kw for kw in keywords if kw in chunk_text]
    #Get words that match and return the ratio (percentage of covered keywords)
    return len(matched) / (len(keywords) or 1)


#The first node in the pipeline
def input_node(state: State):
    query = input("(CTRL+C to quit) Enter your question: ").strip()
    state["messages"].append(HumanMessage(content=query))
    #Takes the input from the user, appends it into the state corresponding to messages,
    #and returns it to the next node: the Retriever
    return state
graph_builder.add_node("input", input_node)


#The second node in the pipeline, responsible for fetching chunks relevant to the user query
#From here on, all time-consuming nodes are timed to measure performance
def retriever_node(state: State):
    start = time.perf_counter()
    query = state["messages"][-1].content
    #Retrieve user query
    query_vector = embeddingModel.encode(query, convert_to_numpy=True).astype("float32")
    query_vector = np.expand_dims(query_vector, axis=0)
    #Vectorize the user query using the embedding model
    distances, indices = index.search(query_vector, kValue)
    chunks = [chunk_texts[i] for i in indices[0] if i<len(chunk_texts)]
    #Retrieve top k similar chunks (similarity determined via distance)
    state["retrieval"] = {
        "chunks": chunks, "distances": distances[0].tolist()
    }
    #Update the state with the chunks
    end = time.perf_counter()
    print(f"RETRIEVER node took {end-start:.1f} seconds")
    return state
graph_builder.add_node("retriever", retriever_node)


#Auxiliary function to reformulate a given query using the selected LLM
def reformulateQuery(query):
    start = time.perf_counter()
    #Construct header of the message
    headers = {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"
    }
    #Construct payload (chosen model and rewriting task with the given query)
    payload = {
        "model": langModel,
        "messages": [
            {"role": "system", "content": "You are an assistant that rewrites user queries into cleaner search queries"},
            {"role": "user", "content": f"Rewrite this question so it works better for searching documents: {query}"}
        ]
    }
    #Post the question and obtain response message 
    resp = requests.post(SITE_URL, json=payload, headers=headers)
    result = resp.json()

    try:
        toResult = result["choices"][0]["message"]["content"].strip()
        print(f"Rewrote question: {toResult}")
        #Extract response (rewrote question)
        end = time.perf_counter()
        print(f"REFORMULATOR took {end-start:.1f} seconds")
        return toResult
    except:
        print("Failed to rewrite query")
        #If error, keep original message
        end = time.perf_counter()
        print(f"REFORMULATOR took {end-start:.1f} seconds")
        return query


#The third node in the pipeline, responsible for deciding if more relevant chunks are required (and fetching them)
def controller_node(state: State):
    #Simple function to vectorize the query
    def encodeQuery(q):
        query_v = embeddingModel.encode(q, convert_to_numpy=True).astype("float32")
        return np.expand_dims(query_v, axis=0)
    start = time.perf_counter()
    query = state["messages"][-1].content
    chunks = state.get("retrieval", {}).get("chunks", [])
    #Extract query and relevant chunks
    coverage = keywordCoverage(query, chunks)
    print(f"Keyword coverage: {coverage:.2f}")
    distances = state.get("retrieval", {}).get("distances", [])
    #Calculate keyword coverage and distances

    #Best distance is used to ensure retrieval quality
    if distances:
        best_distance = min(distances)
    else:
        best_distance = 999
    print(f"Best distance is: {best_distance}")



    #If best distance is high (best chunks' relevance are low), quality is bad -> LLM fallback
    if best_distance > maxDistance:
        #Failed retrieval
        print("Low similarity, now good matches found.")
        state["retrieval"] = {"chunks": [], "distances": []}
        state["retrieval_status"] = "failed"
        state["messages"].append(AIMessage(content="No relevant context could be found."))
        #Update state accordingly, which the LLM node will see
        end = time.perf_counter()
        print(f"CONTROLLER node took {end-start:.1f} seconds")
        return state

    #If keyword coverage is low in the retrieved chunks, more chunks are needed for better context
    elif coverage < minCoverage:
        #Retrieved but low coverage -> fetching more
        print("\nCoverage is low, retrying context retrieval with double the chunks...")
        query_vector = encodeQuery(query)
        distances, indices = index.search(query_vector, kValue*2)
        #Double the number of chunks retrieved - this can be adjusted freely
        newChunks = [chunk_texts[i] for i in indices[0] if i<len(chunk_texts)]
        state["retrieval"] = {
            "chunks": newChunks, "distances": distances[0].tolist()
        }
        #Retrieve the chunks and update state

        newCoverage = keywordCoverage(query, newChunks)
        if newCoverage < minCoverage:
            #Still low coverage, reformulating question and fetching again
            print("\nNot enough coverage! Asking LLM to reformulate query.")
            refined = reformulateQuery(query)
            print(f"Refined query: {refined}")
            #Reformulate the query using the above function to potentially increase coverage

            reformulatedCoverage = keywordCoverage(refined, newChunks)
            if reformulatedCoverage < minCoverage:
                #Low coverage even with reformulated question, failed retrieval
                state["retrieval"] = {"chunks": [], "distances": []}
                state["retrieval_status"] = "failed"
                state["messages"].append(AIMessage(content="No relevant context could be found."))
                #Update state accordingly (fail)
                end = time.perf_counter()
                print(f"CONTROLLER node took {end-start:.1f} seconds")
                return state
            
            #Reformulated question has good coverage, proceed with that
            query_vector = encodeQuery(refined)
            distances, indices = index.search(query_vector, kValue*2)
            reformChunks = [chunk_texts[i] for i in indices[0] if i<len(chunk_texts)]
            #Obtain relevant chunks
            state["retrieval"] = {
                "chunks": reformChunks, "distances": distances[0].tolist()
            }
            state["messages"].append(AIMessage(content=f"Reformulated query used: {refined}"))
            state["retrieval_status"] = "success"
            #Update state accordingly (success)
    
    else:
        #Enough coverage (success)
        state["retrieval_status"] = "success"

    end = time.perf_counter()
    print(f"CONTROLLER node took {end-start:.1f} seconds")
    return state
graph_builder.add_node("controller", controller_node)


#The fourt node in the pipeline, responsible for combining fetched context with the query (RAG) and pass it on to the LLM
def llm_node(state: State):
    start = time.perf_counter()
    query = state["messages"][-1].content
    chunks = state.get("retrieval", {}).get("chunks", [])
    #Retrieve query and chunks from state 
    if not chunks:
        answer = "Could not find relevant information in the document(s). Please try rephrasing the query."
        state["messages"].append(AIMessage(content=answer))
        #No relevant chunks found
        end = time.perf_counter()
        print(f"LLM node took {end-start:.1f} seconds")
        return state
    
    #Else, join the chunks, and formulate a question to the LLM using the user question and the chunks
    context = "\n".join(chunks)
    question = f"Answer this question using the context.\n\nContext:\n{context}\n\nQuestion: {query}"
    #Construct headers and payload just like before, but with the question + chunks
    headers = {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": langModel,
        "messages": [{"role": "user", "content": question}]
    }
    #Obtain response
    resp = requests.post(SITE_URL, json=payload, headers=headers)
    result = resp.json()

    #Try to get answer
    try:
        answer = result["choices"][0]["message"]["content"]
    except:
        answer = f"Error: {result}"
    #Add response message to state
    state["messages"].append(AIMessage(content=answer))
    end = time.perf_counter()
    print(f"LLM node took {end-start:.1f} seconds")
    return state
graph_builder.add_node("llm", llm_node)


#The "last" node in the pipeline (loops back to input), responsible for outputting the answer given by the LLM
#Or if there are any error messages
def output_node(state: State):
    print("\nAnswer: ")
    print(state["messages"][-1].content)
    #Simply print out the message
    return state
graph_builder.add_node("output", output_node)


#Add edges showing the direction of the pipeline (state flow)
graph_builder.add_edge(START, "input")
graph_builder.add_edge("input", "retriever")
graph_builder.add_edge("retriever", "controller")
graph_builder.add_edge("controller", "llm")
graph_builder.add_edge("llm", "output")
graph_builder.add_edge("output", "input")
#From output, loop back to input to allow more chat with the user
graph = graph_builder.compile()
#Compile the graph and initialise the first state
initial_state = State(messages=[])


if __name__ == "__main__":
    try:
        state = initial_state
        while True:
            state = graph.invoke(state)
            #Run the graph
    except KeyboardInterrupt:
        #Handle user wanting to quit, gracefully
        print("\nUser pressed CTRL+C. Exiting gracefully...")
        exit(0)