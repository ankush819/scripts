import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any

# API endpoint
API_URL = "http://localhost:8000"

def format_metrics(metrics: Dict[str, Any]) -> None:
    """Display search metrics in a nice format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Time", f"{metrics['total_time_ms']:.2f}ms")
    with col2:
        st.metric("Search Time", f"{metrics['search_time_ms']:.2f}ms")
    with col3:
        st.metric("Network Time", f"{metrics['network_time_ms']:.2f}ms")

def display_result(result: Dict[str, Any]) -> None:
    """Display a single search result"""
    with st.expander(f"📄 {result['title']}", expanded=False):
        st.markdown("**Content**")
        st.write(result['content'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Category**")
            st.write(result['category'])
        
        with col2:
            st.metric("Distance Score", f"{result['distance']:.4f}")

def main():
    st.set_page_config(
        page_title="Couchbase Vector Search Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Couchbase Vector Search Demo")
    
    # Sidebar
    st.sidebar.title("Search Settings")
    num_results = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # Main search interface
    query = st.text_input("Enter your search query")
    
    if st.button("Search") and query:
        try:
            with st.spinner("Searching..."):
                response = requests.post(
                    f"{API_URL}/search/vector",
                    json={"query": query, "limit": num_results}
                )
                response.raise_for_status()
                data = response.json()
                
                # Display metrics
                st.subheader("📊 Search Metrics")
                format_metrics(data["metrics"])
                
                # Display results
                st.subheader(f"📝 Search Results ({len(data['results'])} found)")
                for result in data["results"]:
                    display_result(result)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 