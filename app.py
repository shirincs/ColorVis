import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import colorsys
import os

# ---- SETUP ----
st.set_page_config(page_title="🎨 Color Genius", layout="centered")

#api_key = os.getenv("API_KEY")
api_key = st.secrets["API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Load and preprocess color data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/fnesh/OneDrive/Documents/Uni/IEN/colors.csv")
    df['rgb_tuple'] = df['rgb'].apply(lambda x: [int(i) for i in x.replace("rgb(", "").replace(")", "").split(",")])
    rgb_array = np.array(df['rgb_tuple'].tolist()) / 255.0
    df['lab'] = list(rgb_array * [100, 1, 1])  # Mock LAB values
    df['standardized_name'] = df['name'].str.lower().replace(" ", "")  # Standardize names
    return df

df = load_data()

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# ---- FUNCTIONS ----
def extract_color_from_text(user_input):
    """Use Gemini API to extract color-related information."""
    prompt = f"""Identify and extract any color name or hex code from the following user input:
    '{user_input}'. Provide ONLY the color name or hex code, without extra explanation."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return None

def find_color(user_input):
    """Find a color based on user input, handling variations and ensuring extraction works."""
    color_value = extract_color_from_text(user_input)
    
    # If Gemini fails, extract last word (may be color name)
    if not color_value:
        words = user_input.lower().split()
        if words:
            color_value = words[-1]  # Last word as fallback (e.g., "teal")

    if not color_value:
        return None
    
    standardized_input = color_value.lower().replace(" ", "")  # Normalize input
    
    if color_value.startswith("#"):
        match = df[df['hex'].str.lower() == color_value.lower()]
    else:
        match = df[df['standardized_name'] == standardized_input]

    return match.iloc[0] if not match.empty else None


def get_similar_colors(target_rgb, n=5):
    """Find similar colors using Euclidean distance."""
    target_lab = np.array(target_rgb) / 255.0 * [100, 1, 1]
    distances = euclidean_distances([target_lab], np.array(df['lab'].tolist()))
    closest_idx = np.argsort(distances)[0][:n]
    return df.iloc[closest_idx]

def adjust_lightness(rgb, factor):
    """Adjust the lightness of an RGB color."""
    r, g, b = [x/255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = max(0.05, min(0.95, l * factor))
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
    return [int(x*255) for x in [new_r, new_g, new_b]]

def get_warm_or_cool_colors(warm=True, n=5):
    """Retrieve warm or cool colors."""
    warm_colors = ['red', 'orange', 'yellow', 'gold', 'brown']
    cool_colors = ['blue', 'cyan', 'green', 'teal', 'purple']
    color_list = warm_colors if warm else cool_colors
    return df[df['name'].str.lower().isin(color_list)].sample(n=min(n, len(df)))

# ---- STREAMLIT UI ----
st.title("🎨 Color Genius")
user_input = st.text_input("Ask about colors (e.g., 'Make crimson darker', 'Find colors like #FF0000')")

if user_input:
    with st.spinner("Thinking..."):
        response = ""
        
        if "warm colors" in user_input.lower():
            colors = get_warm_or_cool_colors(warm=True)
            response = "🔥 Suggested warm colors:\n" + "\n".join([f"- {row['name']} ({row['hex']}, RGB: {row['rgb_tuple']})" for _, row in colors.iterrows()])
        elif "cool colors" in user_input.lower():
            colors = get_warm_or_cool_colors(warm=False)
            response = "❄️ Suggested cool colors:\n" + "\n".join([f"- {row['name']} ({row['hex']}, RGB: {row['rgb_tuple']})" for _, row in colors.iterrows()])
        else:
            color = find_color(user_input)
            
            if color is None:
                response = "❌ Couldn't identify a color. Try a different format."
            else:
                response = f"✅ Found:\n- Name: {color['name']}\n- Hex: {color['hex']}\n- RGB: {color['rgb_tuple']}"
                
                st.markdown(f"""
                <div style='width: 100px; height: 100px; background: {color['hex']}; border: 1px solid #ddd; border-radius: 8px;'></div>
                """, unsafe_allow_html=True)
                
                if "dark" in user_input.lower() or "light" in user_input.lower():
                    factor = 0.7 if "dark" in user_input.lower() else 1.3
                    adjusted = adjust_lightness(color['rgb_tuple'], factor)
                    adjusted_hex = '#%02x%02x%02x' % tuple(adjusted)
                    response += f"\n🎨 Adjusted: {adjusted_hex}"
                    st.markdown(f"""<div style='width:100px; height:100px; background:{adjusted_hex};'></div>""", unsafe_allow_html=True)
                
                if "similar" in user_input.lower() or "like" in user_input.lower():
                    similar = get_similar_colors(color['rgb_tuple'])
                    response += "\n🔍 Similar colors:\n" + "\n".join([f"- {row['name']} ({row['hex']}, RGB: {row['rgb_tuple']})" for _, row in similar.iterrows()])
        
        # Store the interaction in session state
        st.session_state.history.append((user_input, response))

# Display chat history
st.subheader("📜 Chat History")
for user_q, bot_resp in reversed(st.session_state.history): 
    st.markdown(f"**You:** {user_q}")
    st.markdown(f"**Bot:**\n{bot_resp}")
    st.markdown("---")
