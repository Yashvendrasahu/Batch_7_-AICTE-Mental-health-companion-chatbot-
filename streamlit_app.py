import streamlit as st
import os
import pandas as pd
from huggingface_hub import InferenceClient
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="Wellness Companion", page_icon="🧘", layout="wide")

# Token Check
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN missing!")
    st.stop()

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=hf_token)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "tasks" not in st.session_state:
    st.session_state.tasks = {"Drink Water": False, "5-min Walk": False, "Meditation": False}

# --- Sidebar: Features ---
with st.sidebar:
    st.title("🛡️ Wellness Dashboard")
    
    # 1. Daily Wellness Tasks
    st.subheader("Today's Tasks")
    for task in st.session_state.tasks:
        st.session_state.tasks[task] = st.checkbox(task, st.session_state.tasks[task])
    
    # 2. Mood Trend Graph
    if st.session_state.mood_history:
        st.divider()
        st.subheader("Your Mood Journey")
        df = pd.DataFrame(st.session_state.mood_history)
        st.line_chart(df.set_index("Time")["Level"])
    
    # 3. Emergency Help
    st.divider()
    with st.expander("🆘 Need Professional Help?"):
        st.write("Vandrevala Foundation: 9999666555")
        st.write("Aasra Helpline: 9820466726")

# --- Main Interface ---
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🧘 Relaxation Tool", "📝 Journal"])

# --- Tab 1: Chatbot (With Auto Sentiment Analysis) ---
with tab1:
    st.title("🌱 Mental Health Companion")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tell me how you're feeling..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # System Prompt with Auto-Sentiment Request
                system_msg = (
                    "You are a Mental Health Companion. Analyze user sentiment. "
                    "If they sound sad, anxious or lonely, respond with deep empathy and relaxation tips. "
                    "Talk in friendly Hinglish. Keep it natural like a human friend."
                )
                
                messages = [{"role": "system", "content": system_msg}] + \
                           st.session_state.messages[-3:] # Last 3 messages for context

                response = client.chat_completion(messages=messages, max_tokens=300)
                bot_text = response.choices[0].message.content
                st.markdown(bot_text)
                st.session_state.messages.append({"role": "assistant", "content": bot_text})

                # BACKGROUND: Automatic Mood Detection for Graph
                # Hum AI se sirf 1 word mangenge mood update karne ke liye
                try:
                    mood_detect = client.chat_completion(
                        messages=[{"role": "user", "content": f"Analyze this text and return ONLY one word from (Happy, Sad, Angry, Anxious, Neutral): {prompt}"}],
                        max_tokens=10
                    ).choices[0].message.content.strip()
                    
                    mood_val = {"Happy": 5, "Neutral": 4, "Anxious": 3, "Sad": 2, "Angry": 1}.get(mood_detect, 3)
                    st.session_state.mood_history.append({"Time": datetime.now().strftime("%H:%M"), "Level": mood_val})
                except: pass

# --- Tab 2: Relaxation Tool ---
with tab2:
    st.header("🧘 Quick Relaxation")
    st.write("Take a moment to breathe. Follow the steps below:")
    if st.button("Start 1-Minute Breathing"):
        # Simple animation placeholder
        placeholder = st.empty()
        for i in range(3):
            placeholder.info("😤 Breathe In... (4 seconds)")
            # time.sleep(4) -- Hum sleep tab use karenge jab local run karein, 
            # Cloud par delay issues hote hain, isliye simple steps dikhate hain.
            placeholder.success("Hold... (7 seconds)")
            placeholder.warning("😮💨 Breathe Out... (8 seconds)")
        st.balloons()
