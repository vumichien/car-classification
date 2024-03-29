# import torch
# import torch.nn.functional as F
# from torchvision import transforms

from PIL import Image
import numpy as np
from numpy import dot
from numpy.linalg import norm

import onnx, os, time, onnxruntime
import pandas as pd
import threading
# import queue
import cv2
import av

import streamlit as st
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

import args


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_image(x):
    return x.split(', ')[0]


# Transform image to ToTensor
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def transform_image(image, IMG=True):
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.4065), (0.229, 0.224, 0.225)),
    # ])
    if IMG:
        image = np.asarray(Image.open(image))
        # -------------- RESIZE USING CV2 ---------------------
        image = cv2.resize(image, dsize=(224, 224))
        image = np.transpose(image, (2, 0, 1))
        # image = (image/255-np.expand_dims(np.array([0.485, 0.456, 0.4065]),axis = (1,2)))/np.expand_dims(np.array([0.229, 0.224, 0.225]),axis = (1,2))
        image = (image / 255 - np.array(args.MEAN)) / np.array(args.STD)
        img_transformed = np.expand_dims(image.astype(np.float32), axis=0)
        # x = torch.from_numpy(image.astype(np.float32))
        # x = torch.transpose(x, 2, 0) # shape [3, 224, 224]
        # -------------- RESIZE USING CV2 ---------------------
        # img_transformed = []
        # for _ in range(1):
        #     img_transformed.append(x)
        # img_transformed = torch.stack(img_transformed) # shape [1, 3, 224, 224]
    else:
        # -------------- RESIZE USING CV2 ---------------------
        image = cv2.resize(image, dsize=(224, 224))
        image = np.transpose(image, (2, 0, 1))
        # image = (image/255-np.expand_dims(np.array([0.485, 0.456, 0.4065]),axis = (1,2)))/np.expand_dims(np.array([0.229, 0.224, 0.225]),axis = (1,2))
        image = (image / 255 - np.array(args.MEAN)) / np.array(args.STD)
        img_transformed = np.expand_dims(image.astype(np.float32), axis=0)
        # x = torch.from_numpy(image.astype(np.float32))
        # x = torch.transpose(x, 2, 0)
        # -------------- RESIZE USING CV2 ---------------------
        # img_transformed = []
        # img_transformed.append(x)
        # img_transformed = torch.stack(img_transformed)

    return img_transformed


# predict multi-level classification
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_classification(image_tensor, df_train, sub_test_list, embeddings,
                       ort_session, input_name, confidence
                       ):
    # Prediction time
    start = time.time()
    # ort_inputs = {input_name: to_numpy(image_tensor)}
    ort_inputs = {input_name: image_tensor}
    pred, em = ort_session.run(None, ort_inputs)

    if pred.max(axis=1) > confidence:  # threshold to select of item is car part or not, Yes if > 0.5
        # Compute kNN (using Cosine)
        # knn = torch.nn.CosineSimilarity(dim = 1)(torch.tensor(em), embeddings).topk(1, largest=True)

        knn = np.array(
            [dot((em), embeddings[i]) / (norm(em) * norm(embeddings[i])) for i in range(embeddings.shape[0])]).flatten()
        knn = np.argsort(knn)[-1]

        # maker = 'Maker: '+str(df_train.iloc[knn.indices.item(), 0])
        # model = str(df_train.iloc[knn.indices.item(), 1])
        # vehicle = str(df_train.iloc[knn.indices.item(), 2])
        # year = str(df_train.iloc[knn.indices.item(), 3])
        # part = 'Part: '+str(df_train.iloc[knn.indices.item(), 4])
        maker = 'Maker: ' + str(df_train.iloc[knn, 0])
        model = str(df_train.iloc[knn, 1])
        if model == 'nan':
            model = 'Model: No information'
        else:
            model = 'Model: ' + model
        vehicle = str(df_train.iloc[knn, 2])
        if vehicle == 'nan':
            vehicle = 'Vehicle: No information'
        else:
            vehicle = 'Vehicle: ' + vehicle
        year = str(df_train.iloc[knn, 3])
        if year == 'nan':
            year = 'Year: No information'
        else:
            year = 'Year: ' + year
        part = 'Part: ' + str(df_train.iloc[knn, 4])
        predict_time = 'Predict time: ' + str(round(time.time() - start, 4)) + ' seconds'

        # Similarity score
        sim_score = 'Confidence: ' + str(round(pred.max(axis=1).item() * 100, 2)) + '%'

    else:
        maker = 'This is not car part !'
        model = vehicle = year = part = predict_time = sim_score = None

    return {'maker': maker, 'model': model, 'vehicle': vehicle, 'year': year, 'part': part,
            'predict_time': predict_time, 'sim_score': sim_score}


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_classification_frame(image_tensor, df_train, sub_test_list, embeddings,
                             ort_session, input_name
                             ):
    # ort_inputs = {input_name: to_numpy(image_tensor)}
    ort_inputs = {input_name: image_tensor}
    pred, em = ort_session.run(None, ort_inputs)

    if pred.max(axis=1) > args.VIDEO_CONFIDENCE:
        # knn = torch.nn.CosineSimilarity(dim = 1)(torch.tensor(em), embeddings).topk(1, largest=True)
        # part = str(df_train.iloc[knn.indices.item(), 4])
        knn = np.array(
            [dot((em), embeddings[i]) / (norm(em) * norm(embeddings[i])) for i in range(embeddings.shape[0])]).flatten()
        knn = np.argsort(knn)[-1]
        part = str(df_train.iloc[knn, 4])
        # Similarity score
        sim_score = str(round(pred.max(axis=1).item() * 100, 2)) + '%'
    else:
        part = 'No part detected'
        sim_score = ''

    return {'part_name': part, 'sim_score': sim_score}


# predict similarity
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_similarity(image_tensor, df_train, sub_test_list, embeddings,
                   ort_session, input_name
                   ):
    start = time.time()
    # ort_inputs = {input_name: to_numpy(image_tensor)}
    ort_inputs = {input_name: image_tensor}
    pred, em = ort_session.run(None, ort_inputs)

    # Compute kNN (using Cosine)
    # knn = torch.nn.CosineSimilarity(dim = 1)(torch.tensor(em), embeddings).topk(6, largest=True)
    # idx = knn.indices.numpy()
    knn = np.array(
        [dot((em), embeddings[i]) / (norm(em) * norm(embeddings[i])) for i in range(embeddings.shape[0])]).flatten()
    idx = np.argsort(knn)[-6:]
    predict_time = 'Predict time: ' + str(round(time.time() - start, 4)) + ' seconds'
    images_path = 'raw_images'
    images = [os.path.join(images_path, sub_test_list[i]) for i in idx]
    # sub_test_list
    return {'images': images, 'predict_time': predict_time}


# --------------------------------------------------------------------------------------------
# IMAGE INPUT
# --------------------------------------------------------------------------------------------

content_images_dict = {
    name: os.path.join(args.IMAGES_PATH, filee) for name, filee in
    zip(args.CONTENT_IMAGES_NAME, args.CONTENT_IMAGES_FILE)
}


def show_original():
    """ Show Uploaded or Example image before prediction

    Returns:
    -------
    content_file: str
        path to image
    """

    if st.sidebar.checkbox('Upload', value=True, help='Select Upload to browse image from local machine'):
        content_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])
    else:
        content_name = st.sidebar.selectbox("or Choose an example Image below", args.CONTENT_IMAGES_NAME)
        content_file = content_images_dict[content_name]

    col1, col2 = st.columns(2)
    with col1:
        # col1.markdown('## Target image')
        if content_file:
            col1.write('')
            col1.image(content_file, channels='BGR', width=300, clamp=True, caption='Input image')

    return content_file, col2


def image_input(content_file, df_train, sub_test_list, embeddings, ort_session, input_name, col2):
    # Set confidence level
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, args.DEFAULT_CONFIDENCE_THRESHOLD, 0.05,
        help='Choose minimum confidence level. If prediction result below this threshold, no information is shown.'
    )
    if content_file is not None:
        content = transform_image(content_file)
        pred_info = get_classification(
            content, df_train, sub_test_list,
            embeddings, ort_session, input_name, confidence_threshold
        )
        pred_images = get_similarity(
            content, df_train, sub_test_list,
            embeddings, ort_session, input_name
        )

        container = st.container()
        col6, col7 = container.columns([.5, 4])

        with col6:
            if col6.button("PREDICT"):
                print_classification(col2, content_file, pred_info)

        if col7.button("SEARCH SIMILAR"):
            print_classification(col2, content_file, pred_info)

            if pred_info['maker'] != 'This is not car part !':
                # container = st.container()
                print_similar_img(pred_images)  # , container)
            else:
                st.warning("No similar car part image ! Reduce confidence threshold OR Choose another image.")
    else:
        st.success("Upload an Image OR Untick the Upload Button from Options on the sidebar")
        st.info("Navigate input source from Navigation on the sidebar")
        st.stop()


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)


def webcam_input(df_train, sub_test_list, embeddings, ort_session, input_name):
    st.header("Webcam Live Feed")

    class NeuralStyleTransferTransformer(VideoProcessorBase):

        def __init__(self) -> None:
            self._model_lock = threading.Lock()

        def _annotate_image(self, image, pred_info):
            # display the prediction
            part_name = pred_info['part_name']
            confidence = pred_info['sim_score']
            label = f"{part_name} {confidence}"
            cv2.putText(
                image,
                label,
                (2, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 223),
                2,
            )
            return image

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            content = transform_image(image, IMG=False)
            pred_info = get_classification_frame(
                content, df_train, sub_test_list,
                embeddings, ort_session, input_name
            )
            annotated_image = self._annotate_image(image, pred_info)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="live-cassification",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=NeuralStyleTransferTransformer,
        async_processing=True,
    )


def print_classification(col2, content_file, pred_info):
    """ Print classification prediction
    """

    with col2:
        col2.markdown('### Predicted information')
        col2.markdown('')
        if pred_info['maker'] != 'This is not car part !':
            col2.markdown('#### - {}'.format(pred_info['maker']))
            col2.markdown('#### - {}'.format(pred_info['model']))
            col2.markdown('#### - {}'.format(pred_info['vehicle']))
            col2.markdown('#### - {}'.format(pred_info['year']))
            col2.markdown('#### - {}'.format(pred_info['part']))
            col2.markdown('#### - {}'.format(pred_info['predict_time']))
            col2.markdown('#### - {}'.format(pred_info['sim_score']))
        else:
            col2.markdown('### {}'.format(pred_info['maker']))


def print_similar_img(pred_images):
    """ Print similarity images prediction
    """

    st.markdown('### Most similar images')
    st.markdown('#### {}'.format(pred_images['predict_time']))

    col3, col4, col5 = st.columns(3)
    with col3:
        col3.image(pred_images['images'][0], channels='BGR', clamp=True, width=300)
        col3.image(pred_images['images'][1], channels='BGR', clamp=True, width=300)

    with col4:
        # col4.markdown('# ')
        col4.image(pred_images['images'][3], channels='BGR', clamp=True, width=300)
        col4.image(pred_images['images'][4], channels='BGR', clamp=True, width=300)
    with col5:
        col5.image(pred_images['images'][5], channels='BGR', clamp=True, width=300)
        col5.image(pred_images['images'][2], channels='BGR', clamp=True, width=300)
