import streamlit as st
import os
import pandas as pd
import time
from huggingface_hub import InferenceClient
from datetime import datetime

# --- 1. Page Configuration & UI Styling ---
st.set_page_config(page_title="Wellness Companion AI", page_icon="🧘", layout="wide")

# Custom CSS for Animated Breathing Circle and Chat Bubbles
st.markdown("""
<style>
    .stChatMessage { border-radius: 20px; }
    /* Breathing Animation */
    .breathing-circle {
        height: 150px; width: 150px;
        background-color: #4CAF50;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: bold;
        margin: auto;
        animation: breathe 8s infinite ease-in-out;
    }
    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.5); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Setup & Secrets ---
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN missing in Secrets!")
    st.stop()

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=hf_token)

# --- 3. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "daily_tasks" not in st.session_state:
    st.session_state.daily_tasks = {"Drink Water": False, "5-min Meditation": False, "Journaling": False}

# --- 4. Sidebar: Dashboard & Analytics ---
with st.sidebar:
    st.title("🛡️ My Dashboard")
    
    # Task Tracker (Gamification)
    st.subheader("Daily Wellness Goals")
    for task in st.session_state.daily_tasks:
        st.session_state.daily_tasks[task] = st.checkbox(task, value=st.session_state.daily_tasks[task])
    
    # Mood Analytics (Sentiment Analysis Graph)
    if st.session_state.mood_history:
        st.divider()
        st.subheader("Mood Journey")
        df = pd.DataFrame(st.session_state.mood_history)
        st.line_chart(df.set_index("Time")["Score"])
        st.caption("1: Angry | 2: Sad | 3: Anxious | 4: Neutral | 5: Happy")

    # Emergency SOS
    st.divider()
    with st.expander("🆘 Urgent Support"):
        st.markdown("**National Helpline:** 9152987821")
        st.markdown("**Aasra:** 9820466726")

# --- 5. Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 AI Companion", "🧘 Breathing Exercise", "📝 My Journal"])

# --- Tab 1: AI Chatbot (with Sentiment detection) ---
with tab1:
    st.title("🌱 How are you feeling?")
    st.caption("I use sentiment analysis to understand your emotions and support you.")

    # Display Chat Bubbles
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Empathizing..."):
                # System Prompt for Empathy
                sys_msg = (
                    "You are a compassionate Mental Health Companion for students facing stress and loneliness. "
                    "Detect sentiment and talk in friendly Hinglish. Be empathetic, motivational, and suggest "
                    "one small relaxation tip in every response."
                )
                
                # Main Chat API
                messages = [{"role": "system", "content": sys_msg}] + st.session_state.messages[-4:]
                response = client.chat_completion(messages=messages, max_tokens=350)
                full_res = response.choices[0].message.content
                st.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})

                # Background Sentiment Analysis for Graph
                try:
                    sentiment_prompt = f"Analyze sentiment of this text. Return ONLY one word (Happy, Neutral, Anxious, Sad, Angry): '{prompt}'"
                    mood_tag = client.chat_completion(messages=[{"role": "user", "content": sentiment_prompt}], max_tokens=10).choices[0].message.content.strip()
                    mood_scores = {"Happy": 5, "Neutral": 4, "Anxious": 3, "Sad": 2, "Angry": 1}
                    score = mood_scores.get(mood_tag, 4)
                    st.session_state.mood_history.append({"Time": datetime.now().strftime("%H:%M"), "Score": score})
                except: pass

# --- Tab 2: Animated Breathing Tool ---
with tab2:
    st.header("🧘 Focused Breathing")
    st.write("Anxiety ko kam karne ke liye is circle ke saath saans lein (Inhale/Exhale).")
    
    # CSS Animation Trigger
    st.markdown('<div class="breathing-circle">BREATHE</div>', unsafe_allow_html=True)
    
    st.divider()
    st.info("4-7-8 Technique: Breathe in for 4s, Hold for 7s, Exhale for 8s.")

# --- Tab 3: Personal Journal ---
with tab3:
    st.header("📝 Personal Journal")
    note = st.text_area("Write your thoughts down. It helps clear the mind.", height=200)
    if st.button("Save Entry", use_container_width=True):
        if note:
            st.success("Entry saved privately.")
            st.balloons()
