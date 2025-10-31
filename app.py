import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="ChessLens",
    page_icon="♟️",
    layout="centered"
)

# Constants
BOARD_SIZE = 400  # Size to crop the board to

def detect_and_crop_chessboard(image):
    # Convert PIL image to OpenCV format (BGR)
    img_array = np.array(image)
    
    # Ensure image is in BGR format for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else: # Handle grayscale or other formats
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR) if len(img_array.shape) == 2 else img_array

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Add flags to make detection more robust
    # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    
    # Find the 7x7 internal corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), flags=None)
    
    if not ret:
        return None

    # --- Extrapolation Logic ---
    # Estimate the average square size from the inner grid
    # Horizontal distance (top row of inner corners)
    h_dist = np.linalg.norm(corners[0][0] - corners[6][0]) / 6.0
    # Vertical distance (left column of inner corners)
    v_dist = np.linalg.norm(corners[0][0] - corners[42][0]) / 6.0

    # Get the four outermost inner corners
    top_left_inner = corners[0][0]
    top_right_inner = corners[6][0]
    bottom_left_inner = corners[42][0]
    
    # Define the direction vectors based on the grid orientation
    # Vector pointing right
    vec_right = (top_right_inner - top_left_inner) / 6.0
    # Vector pointing down
    vec_down = (bottom_left_inner - top_left_inner) / 6.0

    # Extrapolate to find the four outer corners of the 8x8 board
    # Move one step left and one step up from the top-left inner corner
    top_left_outer = bottom_left_inner + (vec_right * 7) + vec_down
    # Move one step right and one step up from the top-right inner corner
    top_right_outer = bottom_left_inner - vec_right + vec_down
    # Move one step left and one step down from the bottom-left inner corner
    bottom_left_outer = top_right_inner + vec_right - vec_down
    # Move one step right and one step down from the bottom-right inner corner
    bottom_right_outer = top_left_inner - vec_right - vec_down

    # Define source and destination points for the full board
    src_pts = np.float32([top_left_outer, top_right_outer, bottom_left_outer, bottom_right_outer])
    
    dst_pts = np.float32([
        [0, 0],
        [BOARD_SIZE, 0],
        [0, BOARD_SIZE],
        [BOARD_SIZE, BOARD_SIZE]
    ])
    
    # Get perspective transform matrix and warp the image
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_board = cv2.warpPerspective(img_bgr, matrix, (BOARD_SIZE, BOARD_SIZE))
    
    return cropped_board


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
        # Display the image
        image = Image.open(uploaded_file)
        
        st.subheader("Uploaded Image")
        st.image(image, caption=f"Original image: {uploaded_file.name}", use_container_width=True)
        
        # Detect and crop the chessboard
        with st.spinner("Detecting chessboard..."):
            cropped_board = detect_and_crop_chessboard(image)
        
        if cropped_board is not None:
            st.success("✅ Chessboard detected successfully!")
            
            # Display the cropped board
            st.subheader("Detected Chessboard")
            # Convert BGR to RGB for display
            cropped_rgb = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2RGB)
            st.image(cropped_rgb, caption=f"Cropped board ({BOARD_SIZE}x{BOARD_SIZE})", width=400)
            
            # Placeholder for FEN output
            st.subheader("FEN Output")
            st.text_input(
                "FEN Notation:",
                value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                disabled=True,
                help="This is a placeholder."
            )
        else:
            st.error("Could not detect a chessboard in the image.")
            st.info("""
            """)
    
    else:
        # Show helpful message when no image is uploaded
        st.markdown("""
        
        """)


if __name__ == "__main__":
    main()
