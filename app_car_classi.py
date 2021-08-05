import streamlit as st

from utils import *
import shutil
import time, os, json, torch, onnx, onnxruntime
from PIL import Image
import pandas as pd
#import cv2
from collections import deque, Counter
import args

st.set_page_config(
    page_title=args.PAGE_TITLE,
    page_icon=args.PAGE_ICON, layout=args.LAYOUT, initial_sidebar_state='auto'
)

st.title(args.LANDINGPAGE_TITLE)
st.sidebar.title(args.SIDEBAR_TITLE)
method = st.sidebar.radio('Choose input source ->', options=['Image','Webcam'])

# Load model
df_train = pd.read_csv('full_set.csv')
sub_test_list = list(df_train['Image'].map(lambda x: get_image(x)))
embeddings = torch.load('embeddings.pt')
PATH = 'model_onnx.onnx'
ort_session = onnxruntime.InferenceSession(PATH)
input_name = ort_session.get_inputs()[0].name

if method == 'Image':
    st.sidebar.header('Options')
    content_file, col2 = show_original()
    image_input(
        content_file, df_train, sub_test_list, embeddings, ort_session, input_name, col2
    )
else:
    webcam_input(
        df_train, sub_test_list, embeddings, ort_session, input_name
    )
