import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List

st.set_page_config(page_title="Vector Search Demo", layout="wide")

def format_metrics(metrics: Dict) -> None:
    """Display performance metrics in a clean format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latency Breakdown")
        latency_df = pd.DataFrame({
            'Metric': ['Total Time', 'Embedding Time', 'Search Time', 'Network Time'],
            'Duration (ms)': [
                f"{metrics['total_time_ms']:.2f}",
                f"{metrics['embedding_time_ms']:.2f}",
                f"{metrics['search_time_ms']:.2f}",
                f"{metrics['network_time_ms']:.2f}"
            ]
        })
        st.dataframe(latency_df, hide_index=True)
    
    with col2:
        st.subheader("Search Quality")
        st.metric("Recall Score", f"{metrics['recall']*100:.1f}%")

def display_results(results: List[Dict]) -> None:
    """Display search results in a clean format"""
    st.subheader(f"Found {len(results)} results")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"{i}. {result['title']}", expanded=i==1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Content:**")
                st.write(result['content'])
            
            with col2:
                st.markdown("**Category:**")
                st.write(result['category'])
                st.markdown("**Distance Score:**")
                st.write(f"{result['distance']:.4f}")
        st.divider()

def main():
    st.title("📚 News Article Search")
    
    # Sidebar for search configuration
    with st.sidebar:
        st.header("Search Configuration")
        search_type = st.selectbox(
            "Select Search Type",
            options=["semantic", "hybrid"],
            format_func=lambda x: "Semantic Search" if x == "semantic" else "Hybrid Search"
        )
        
        num_results = st.slider("Number of Results", min_value=1, max_value=10, value=5)
        
        st.markdown("""
        ### Search Types
        - **Semantic Search**: Uses vector similarity to find related articles
        - **Hybrid Search**: Combines vector similarity with keyword matching
        """)
    
    # Main search interface
    query = st.text_input("Enter your search query", placeholder="e.g., latest technology innovations in AI")
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    "http://localhost:8000/search",
                    json={
                        "query": query,
                        "search_type": search_type,
                        "limit": num_results
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Display metrics
                st.header("Search Metrics")
                format_metrics(data['metrics'])
                
                # Display results
                st.header("Search Results")
                display_results(data['results'])
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to search service: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 