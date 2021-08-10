
#import shutil
#import cv2
#from PIL import Image
#from collections import deque, Counter

import time, os, json, onnx, onnxruntime
#import torch
import pandas as pd
import streamlit as st

from utils import *
import args


st.set_page_config(
    page_title=args.PAGE_TITLE,
    page_icon=args.PAGE_ICON, layout=args.LAYOUT, initial_sidebar_state='auto'
)
hide_streamlit_style = """
    <style>
    footer {
    visibility: hidden;
        }
    footer:after {
        content:'© 2021 Vu Minh Chien';
        visibility: visible;
        display: block;
        position: relative;
        #background-color: red;
        padding: 5px;
        top: 2px;
            }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title(args.LANDINGPAGE_TITLE)
st.sidebar.title(args.SIDEBAR_TITLE)
method = st.sidebar.radio('Choose input source 👇', options=['Image', 'Webcam'])

# Load model
@st.cache(suppress_st_warning=False)
def initial_setup():
    df_train = pd.read_csv('full_set.csv')
    sub_test_list = sorted(list(df_train['Image'].map(lambda x: get_image(x))))
    # embeddings = torch.load('embeddings.pt')
    with open('embeddings.npy', 'rb') as f:
        embeddings = np.load(f)
    PATH = 'model_onnx.onnx'
    ort_session = onnxruntime.InferenceSession(PATH)
    input_name = ort_session.get_inputs()[0].name
    return df_train, sub_test_list, embeddings, ort_session, input_name

df_train, sub_test_list, embeddings, ort_session, input_name = initial_setup()

if method == 'Image':
    st.sidebar.markdown('---')
    st.sidebar.header('Options')
    content_file, col2 = show_original()
    image_input(
        content_file, df_train, sub_test_list, embeddings, ort_session, input_name, col2
    )
else:
    webcam_input(
        df_train, sub_test_list, embeddings, ort_session, input_name
    )
