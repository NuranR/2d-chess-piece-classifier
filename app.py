import streamlit as st
from PIL import Image
import numpy as np
import cv2
from streamlit_cropper import st_cropper
import tensorflow as tf
from pathlib import Path
import chess
import chess.svg

# Page configuration 
st.set_page_config(
    page_title="ChessLens",
    page_icon="♟️",
    layout="wide"
)

# Constants
BOARD_SIZE = 400
SQUARE_SIZE = 50 
MODEL_PATH = "models/piece_classifier_model.tflite"

PIECE_MAP = {
    0: '1',   # Empty square
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',  # White pieces
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'  # Black pieces
}

@st.cache_resource
def load_model():
    """Load the TFLite model (cached for performance)"""
    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found at `{MODEL_PATH}`")
        return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_squares(board_image):
    # Convert PIL to numpy array and to grayscale
    img_array = np.array(board_image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    squares = []
    for row in range(8):
        for col in range(8):
            y_start = row * SQUARE_SIZE
            y_end = (row + 1) * SQUARE_SIZE
            x_start = col * SQUARE_SIZE
            x_end = (col + 1) * SQUARE_SIZE
            
            square = gray[y_start:y_end, x_start:x_end]
            squares.append(square)
    
    return squares

def predict_board(interpreter, input_details, output_details, squares):
    predictions = []
    
    for square in squares:
        # Preprocess: normalize and reshape
        img_array = square.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predicted class
        predicted_class = np.argmax(output_data)
        predictions.append(predicted_class)
    
    return predictions

def predictions_to_fen(predictions):
    fen = ""
    
    for row in range(8):
        empty_count = 0
        row_preds = predictions[row * 8:(row + 1) * 8]
        
        for pred in row_preds:
            piece = PIECE_MAP.get(pred, '1')
            
            if piece == '1':  # Empty square
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += piece
        
        # Add remaining empty squares
        if empty_count > 0:
            fen += str(empty_count)
        
        # Add rank separator (except after last rank)
        if row < 7:
            fen += '/'
    
    return fen


def build_full_fen(base_fen, turn):
    """Build complete FEN with turn and default values"""
    # Default values: KQkq castling, no en passant, 0 halfmove, 1 fullmove
    full_fen = f"{base_fen} {turn} KQkq - 0 1"
    return full_fen


def get_lichess_editor_url(fen_string):
    import urllib.parse
    
    encoded_fen = urllib.parse.quote(fen_string)
    
    lichess_url = f"https://lichess.org/editor/{encoded_fen}"
    
    return lichess_url


def render_chess_board(fen_string):
    try:
        board = chess.Board(fen_string)
        svg_board = chess.svg.board(
            board=board,
            size=400,
            coordinates=True,
            colors={
                "square light": "#f0d9b5",
                "square dark": "#b58863",
                "margin": "#212121"
            }
        )
        return svg_board
    except Exception as e:
        st.error(f"Error rendering board: {e}")
        return None


def main():
    st.title("♟️ ChessLens")
    st.caption("Extract FEN notation from chess board images")
    
    # Initialize session state for extracted FEN
    if 'base_fen' not in st.session_state:
        st.session_state.base_fen = None
    
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    interpreter, input_details, output_details = model_data

    st.markdown("""
        <style>
        /* Main container shrinks to fit content */
        [data-testid="stFileUploader"] {
            width: max-content;
        }
        /* The actual dropzone - make it transparent and borderless */
        [data-testid="stFileUploader"] section {
            padding: 0;
            float: left;
            background-color: transparent !important;
            border: none !important;
        }
        /* Hides the "Drag and drop file here" text/icon */
        [data-testid="stFileUploader"] section > input + div {
            display: none;
        }
        
        </style>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload chess board image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        
        # Resize image to max height of 400px while maintaining aspect ratio
        max_height = 400
        aspect_ratio = original_image.width / original_image.height
        if original_image.height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            display_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            display_image = original_image
        
        # Initialize session state for cropping
        if 'crop_mode' not in st.session_state:
            st.session_state.crop_mode = False
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            if st.button("✂️ Enable Crop Tool", help="Click to enable cropping - drag corners to select the board area"):
                st.session_state.crop_mode = True

            cropped_img = st_cropper(
                display_image, 
                realtime_update=True,
                box_color='#00FF00' if st.session_state.crop_mode else '#00000000',  # Transparent when not in crop mode
                aspect_ratio=None,
                return_type='box' if not st.session_state.crop_mode else 'image'  # Don't crop unless in crop mode
            )
            
            # Determine which image to use
            if st.session_state.crop_mode and cropped_img is not None:
                final_image = cropped_img
            else:
                final_image = display_image
            
            # Show the image to be processed
            if final_image is not None:
                resized_board = final_image.resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.LANCZOS)
                
                # Turn selection and Extract button
                turn = st.radio(
                    "Select turn:",
                    options=["w", "b"],
                    format_func=lambda x: "⚪ White to move" if x == "w" else "⚫ Black to move",
                    horizontal=True,
                    key="turn_selector"
                )
                
                # Extract FEN button
                if st.button("🎯 Extract FEN", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyzing chess pieces..."):
                        squares = extract_squares(resized_board)
                        predictions = predict_board(interpreter, input_details, output_details, squares)
                        base_fen = predictions_to_fen(predictions)
                        
                        # Build complete FEN with turn
                        full_fen = build_full_fen(base_fen, turn)
                        st.session_state.base_fen = full_fen
                    
                    st.toast("✨ FEN extracted successfully!", icon="✅")
        
        with col_right:
            st.subheader("Preview")
            if st.session_state.base_fen is not None:
                svg_board = render_chess_board(st.session_state.base_fen)
            else:
                svg_board = render_chess_board("8/8/8/8/8/8/8/8 w KQkq - 0 1")
            
            if svg_board:
                st.markdown(
                    f'<div style="display: flex; justify-content: center;">{svg_board}</div>',
                    unsafe_allow_html=True
                )
            st.divider()
            # Display results if FEN has been extracted
            if st.session_state.base_fen is not None:
                    st.code(st.session_state.base_fen, language=None)
                    
                    lichess_url = get_lichess_editor_url(st.session_state.base_fen)
                    
                    # Links to external tools
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### 🔗 [Lichess Editor]({lichess_url})")
                    
                    with col2:
                        analysis_url = lichess_url.replace("/editor/", "/analysis/")
                        st.markdown(f"### 📊 [Lichess Analysis]({analysis_url})")
        
    else:
        # Reset session state when no file is uploaded
        st.session_state.base_fen = None
        st.session_state.crop_mode = False
        # Show helpful message when no image is uploaded
        st.markdown("""
        ### 👋 Welcome to ChessLens!
                
        Upload an image of a chess board to get started.
        """)

if __name__ == "__main__":
    main()