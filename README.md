---
title: ChessLens
emoji: ♟️
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
---

# ♟️ ChessLens - Chess Position to FEN Converter

Convert chess board images to FEN notation using deep learning! Upload a photo of any chess position and get the FEN string instantly.

## 🎯 Features

- 📸 **Image Upload** - Support for JPG, PNG formats
- ✂️ **Manual Cropping** - Interactive tool to select just the board
- 🤖 **AI Piece Recognition** - TensorFlow Lite model trained on 512K samples
- 📋 **Complete FEN Generation** - Including turn, castling rights, en passant
- ♟️ **Visual Board Display** - Beautiful SVG chess board rendering
- 🔗 **Lichess Integration** - Direct links to editor and analysis board

## 🚀 How to Use

1. Upload an image of a chess position
2. Crop to select only the chess board
3. Click "Extract FEN" to analyze
4. Configure game state (turn, castling, etc.)
5. Copy the FEN or open in Lichess for analysis

Built with Streamlit, OpenCV, and TensorFlow Lite.
