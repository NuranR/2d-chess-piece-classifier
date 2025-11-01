import streamlit as st
from PIL import Image
import numpy as np
import cv2
from streamlit_cropper import st_cropper
import tensorflow as tf
from pathlib import Path
import chess
import chess.svg

## Page configuration
st.set_page_config(
    page_title="ChessLens",
    page_icon="♟️",
    layout="centered"
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


def get_full_fen_dialog(base_fen):
    st.subheader("Board Configuration")
    st.info("Configure additional details for the full FEN notation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        turn = st.radio(
            "Active player:",
            options=["w", "b"],
            format_func=lambda x: "White" if x == "w" else "Black",
            horizontal=True
        )
        
        castling = st.text_input(
            "Castling rights:",
            value="KQkq",
            help="K=White kingside, Q=White queenside, k=Black kingside, q=Black queenside. Use '-' for none"
        )
    
    with col2:
        en_passant = st.text_input(
            "En passant target:",
            value="-",
            help="Square where en passant capture is possible (e.g., 'e3') or '-' for none"
        )
        
        halfmove = st.number_input(
            "Halfmove clock:",
            min_value=0,
            value=0,
            help="Number of halfmoves since last capture or pawn advance"
        )
    
    fullmove = st.number_input(
        "Fullmove number:",
        min_value=1,
        value=1,
        help="Number of full moves (starts at 1, increments after Black's move)"
    )
    
    # Construct full FEN
    full_fen = f"{base_fen} {turn} {castling} {en_passant} {halfmove} {fullmove}"
    
    return full_fen


def get_lichess_editor_url(fen_string):
    import urllib.parse
    
    # URL encode the FEN string
    encoded_fen = urllib.parse.quote(fen_string)
    
    # Lichess board editor URL
    lichess_url = f"https://lichess.org/editor/{encoded_fen}"
    
    return lichess_url


def render_chess_board(fen_string):
    try:
        board = chess.Board(fen_string)
        # Generate SVG
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
    # Header
    st.title("♟️ ChessLens")
    
    # Load model at startup
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    interpreter, input_details, output_details = model_data
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a chessboard"
    )
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        
        st.info("Drag the corners to select just the board")
        
        # Interactive cropper
        cropped_img = st_cropper(
            image, 
            realtime_update=True,
            box_color='#00FF00',
            aspect_ratio=None,  # Allow free cropping
        )
        
        # Show the cropped result
        if cropped_img is not None:       
            # Resize to standard size (400x400)
            resized_board = cropped_img.resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.LANCZOS)
            # st.image(resized_board, caption=f"Resized to {BOARD_SIZE}x{BOARD_SIZE}", width=400)
            
            # Extract FEN            
            if st.button("Extract FEN", type="primary"):
                with st.spinner("Analyzing chess pieces..."):
                    squares = extract_squares(resized_board)
                    predictions = predict_board(interpreter, input_details, output_details, squares)
                    fen_string = predictions_to_fen(predictions)
                
                st.success("FEN extracted successfully!")
                
                # Get full FEN with additional details
                st.divider()
                full_fen = get_full_fen_dialog(fen_string)
                
                st.divider()
                st.subheader("Complete FEN Notation")
                st.code(full_fen, language=None)
                
                # Chess Board Visualization
                st.subheader("Chess Board")
                svg_board = render_chess_board(full_fen)
                
                if svg_board:
                    # Display SVG board
                    st.markdown(
                        f'<div style="display: flex; justify-content: center;">{svg_board}</div>',
                        unsafe_allow_html=True
                    )
                
                # Get Lichess URL for analysis
                lichess_url = get_lichess_editor_url(full_fen)
                
                st.divider()
                
                # Links to external tools
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### 🔗 [Open in Lichess Editor]({lichess_url})")
                
                with col2:
                    analysis_url = lichess_url.replace("/editor/", "/analysis/")
                    st.markdown(f"### 📊 [Open in Lichess Analysis]({analysis_url})")
    
    else:
        # Show helpful message when no image is uploaded
        st.markdown("""
        Upload an image of a chess board to get started!
        """)


if __name__ == "__main__":
    main()