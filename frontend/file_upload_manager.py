"""
File Upload Manager - Simplified version for MinIO object storage
Only keeps essential functionality without volume-based operations.
"""

import streamlit as st
from typing import List
from datetime import datetime


class FileUploadManager:
    """
    Simplified upload manager for MinIO object storage.
    Volume-based upload methods have been removed.
    """
    
    def __init__(self):
        """Initialize the upload manager"""
        self.upload_history = []
    
    def get_file_info(self, uploaded_files: List[any]) -> dict:
        """
        Get information about uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Dictionary with file information
        """
        if not uploaded_files:
            return {"count": 0, "total_size": 0}
        
        total_size = sum(len(file.getvalue()) for file in uploaded_files)
        
        return {
            "count": len(uploaded_files),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files": [
                {
                    "name": file.name,
                    "size": len(file.getvalue()),
                    "size_mb": len(file.getvalue()) / (1024 * 1024)
                }
                for file in uploaded_files
            ]
        }
    
    def display_file_summary(self, uploaded_files: List[any]) -> None:
        """
        Display a summary of uploaded files in Streamlit.
        
        Args:
            uploaded_files: List of uploaded file objects
        """
        if not uploaded_files:
            st.info("No files selected")
            return
        
        file_info = self.get_file_info(uploaded_files)
        
        st.write(f"**{file_info['count']} files selected ({file_info['total_size_mb']:.2f} MB)**")
        
        with st.expander("Show selected files"):
            for file_data in file_info['files']:
                st.write(f"- {file_data['name']} ({file_data['size_mb']:.2f} MB)")

