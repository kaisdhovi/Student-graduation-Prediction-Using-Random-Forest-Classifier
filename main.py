"""
@author kelompok 2
"""
from model import RandomHutanClassifier, Decision_Node, QuestionSplit, Leaf
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import base64


pickle_in = open("rfc_88.11%.pkl", "rb")
classifier = pickle.load(pickle_in)

def main():
    st.set_page_config(page_title="Student Performance ML App", page_icon="🎓", layout="centered")

    def add_background(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        css = f"""
        <style>
        .block-container {{
            padding: 0 !important;
        }}
        header {{
            display: none;
        }}
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            width: 100vw;
            margin: 0;
            padding: 0;
        }} 
        header, footer {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}
        </style>
        
        """
        st.markdown(css, unsafe_allow_html=True)
        
        

    # Manage pages with session state
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "positive" not in st.session_state:
        st.session_state.positive = False

    def next_page():
        st.session_state.page += 1

    def prev_page():
        st.session_state.page -= 1

    if st.session_state.page == 1:
        add_background("bg1.png")
        st.markdown(
        """
        <style>
        .stButton button {
            background-color:#000000; 
            color: white;
            font-size: 16px;
            border-radius: 12px;
            padding: 20px 50px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #ffffff; /* Warna tombol saat hover */
            color: #000; /* Warna teks saat hover */
            transform: scale(1.05); /* Efek zoom saat hover */
        }
        .stButton {
            position: fixed;
            bottom: 60px;
            left: 140px;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True)
    
        if st.button("Start Your Detection Now"):
            next_page()
    # Page 2: Input Course, Application Mode, Tuition Fee, Age
    elif st.session_state.page == 2:
        add_background("bg2.png")
        st.write("")
        st.write("")

        st.subheader("fill in the form below to predict student performance")

        col1, col2, col3 = st.columns(3, gap="small")

    # Column 1 inputs
        with col1:
            co_enc = [
                "Biofuel Production Technologies",
                "Animation and Multimedia Design",
                "Social Service (evening attendance)",
                "Agronomy",
                "Communication Design",
                "Veterinary Nursing",
                "Informatics Engineering",
                "Equinculture",
                "Management",
                "Social Service",
                "Tourism",
                "Nursing",
                "Oral Hygiene",
                "Advertising and Marketing Management",
                "Journalism and Communication",
                "Basic Education",
                "Management (evening attendance)"
            ]
            st.session_state.course = st.selectbox("Choose Course Name", co_enc)

            am_enc = [
                "1st phase - general contingent",
                "2nd phase - general contingent",
                "3rd phase - general contingent",
                "Holders of other higher courses",
                "Over 23 years old",
                "Transfer",
                "Change of course",
                "Technological specialization diploma holders",
                "Others"
            ]
            st.session_state.am_uc = st.selectbox("Choose Application Mode", am_enc)

            fee_co = ["YES", "NO"]
            st.session_state.fee_uc = st.selectbox("Are you updating your tuition fee?", fee_co)
            st.session_state.age = st.number_input('Choose Age at Enrollment:', min_value=17, max_value=70)


        # Column 2 inputs
        with col2:
            ev1 = st.number_input('1st Sem (Evaluations):', min_value=0, max_value=50)
            ap1 = st.number_input('1st Sem (Approved):', min_value=0, max_value=30)
            gr1 = st.number_input('1st Sem (Grade: 0-20):', min_value=0.0, max_value=20.0, step=0.001)

        # Column 3 inputs
        with col3:
            ev2 = st.number_input('2nd Sem (Evaluations):', min_value=0, max_value=50)
            ap2 = st.number_input('2nd Sem (Approved):', min_value=0, max_value=30)
            gr2 = st.number_input('2nd Sem (Grade: 0-20):', min_value=0.0, max_value=20.0, step=0.001)

        # Submit button
        st.markdown(
            """
            <style>
            .stButton button {
                background-color:#000000; 
                color: white;
                font-size: 16px;
                border-radius: 12px;
                padding: 10px 20px;
                border: none;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #ffffff; /* Warna tombol saat hover */
                color: #000; /* Warna teks saat hover */
                transform: scale(1.05); /* Efek zoom saat hover */
            }
            </style>
            """,
            unsafe_allow_html=True)

        if st.button("Submit"):
            feat_values = {
                "Course": st.session_state.course,
                "Application Mode": st.session_state.am_uc,
                "Fee update": st.session_state.fee_uc,
                "Age": st.session_state.age,
                "Curricular units 1st sem (evaluations)": ev1,
                "Curricular units 1st sem (approved)": ap1,
                "Curricular units 1st sem (grade: 0-20)": gr1,
                "Curricular units 2nd sem (evaluations)": ev2,
                "Curricular units 2nd sem (approved)": ap2,
                "Curricular units 2nd sem (grade: 0-20)": gr2
            }
            new_df = pd.DataFrame([feat_values])
            predictions = classifier.predict(new_df)

            if predictions[0] == 'Dropout':
                st.session_state.positive = 1
            elif predictions[0] == 'Graduate':
                st.session_state.positive = 0
            
            next_page()
    
    # Page 3: Output Prediction
    elif st.session_state.page == 3:
        if st.session_state.positive ==1:
            add_background("do.png")
            st.markdown(
        """
        <style>
        .stButton button {
            background-color:#000000; 
            color: white;
            font-size: 16px;
            border-radius: 12px;
            padding: 10px 50px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #ffffff; /* Warna tombol saat hover */
            color: #000; /* Warna teks saat hover */
            transform: scale(1.05); /* Efek zoom saat hover */
        }
        .stButton {
            position: fixed;
            bottom: 80px;
            left: 250px;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True)
            if st.button("Back"):
                st.session_state.page = 1
        else:
            add_background("graduate.png")
            st.markdown(
        """
        <style>
        .stButton button {
            background-color:#000000; 
            color: white;
            font-size: 16px;
            border-radius: 12px;
            padding: 10px 50px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #ffffff; /* Warna tombol saat hover */
            color: #000; /* Warna teks saat hover */
            transform: scale(1.05); /* Efek zoom saat hover */
        }
        .stButton {
            position: fixed;
            bottom: 80px;
            left: 250px;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True)
            if st.button("Back"):
                st.session_state.page = 1
if __name__ == '__main__':
    main()