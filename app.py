import streamlit as st
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="ChessLens",
    page_icon="♟️",
    layout="centered"
)

def main():
    # Header
    st.title("♟️ ChessLens")
    st.markdown("Upload a chessboard screenshot and get the FEN notation!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chessboard image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a chessboard"
    )
    
    # Display the uploaded image
    if uploaded_file is not None:
        # st.success(f"File uploaded: {uploaded_file.name}")
        
        # Display the image
        image = Image.open(uploaded_file)
        
        st.subheader("Uploaded Image")
        st.image(image, caption=f"Original image: {uploaded_file.name}", use_container_width=True)
                
        # Placeholder for FEN output
        st.subheader("FEN Output")
        
        # Placeholder for where FEN will be displayed
        st.text_input(
            "FEN Notation:",
            value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            disabled=True,
            help="This is a placeholder. We'll extract the real FEN in the next step."
        )
    
    else:
        # Show helpful message when no image is uploaded
        st.markdown("""
        
        """)


if __name__ == "__main__":
    main()
