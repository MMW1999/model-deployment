# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#pip install ecg-plot



#import ecg_plot
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import scipy.io

streamlit run deployment_model.py

#---------------------------------#
st.title('12-lead ECG evaluation')


#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
    page_title='🫀 ECG Classification',
    # anatomical heart favicon
    page_icon="https://api.iconify.design/openmoji/anatomical-heart.svg?width=500",
    layout='wide'
)

# Page Intro
st.write("""
# 12-lead ECG Classification

For this webbased application, a Residual neural network was trained on 12 leads ECG data
to recognize pathologies from the WOBBLER acronym on the ECG. In addition, the model is also able 
to detect arrhytmias such as ventricle fibrillation as atrial fibrillation 
(a full list of the possible predictions can be found in the supplementary materials on the [github page](https://github.com/MMW1999/ECG_arrhythmia)).

This project was created as part of an technical medicine internship by Margot van Hest (2024)

-------
""".strip())

#---------------------------------#
# Data preprocessing and Model building

@st.cache_data
def load_data(filename):
  filename.seek(0)      #find beginning of ECG data
  x = scipy.io.loadmat(filename)
  x = x['val'][0]
  uploaded_ecg = np.array([x])
  return uploaded_ecg


@st.cache_data
def preprocess_ecg(uploaded_ecg):
    """
    Preprocesses ECG signal into fragments of specified length and sampling rate.
    
    Parameters:
    - ecg_signal: numpy array, the ECG signal to be preprocessed.
    
    Returns:
    - fragments: list of numpy arrays, preprocessed ECG signal fragments.
    """

    fs =500             #sampling frequency of ECG
    fragment_length = 10*fs      # maximum length of ECG fragment as input for model
    overlap = 0.25      # overlap of ECG fragments fed to model

    # remove NaN from signal and center the data around zero
    uploaded_ecg = np.nan_to_num(uploaded_ecg)
    uploaded_ecg = uploaded_ecg - np.mean(uploaded_ecg)

    # Calculate the number of fragments with the specified overlap
    stride = int(fragment_length * (1 - overlap))
    total_samples = len(uploaded_ecg[0])
    num_fragments = (total_samples - fragment_length) // stride + 1

    fragments = []

    for i in range(num_fragments):
        start_index = i*stride
        end_index = start_index + fragment_length

        # for last fragment if its shorter than the fragment length this 
        # fragment needs to be padded to have the right input size for the model
        # which is why the following check is implemented
        if end_index - start_index < fragment_length:
          # Calculate the length of the padding
          padding_length = fragment_length - (end_index - start_index)
          # Pad the fragment with zeros along the time axis
          fragment = np.pad(uploaded_ecg[start_index:end_index, :], 
                            ((0, 0), (0, padding_length)), mode='constant')
        else:
          fragment = uploaded_ecg[start_index:end_index,:]
      
        fragments.append(fragment)

        # stack fragment into signle numpy array
        fragments_stacked = np.stack(fragments)

        return fragments_stacked


model_path = 'models/deployment_model.hdf5'
classes = ['Normal','Atrial Fibrillation','Other','Noise']

@st.cache_resource
def get_model(model_path):
    model = load_model(f'{model_path}')
    return model

@st.cache_resource
def get_prediction(data,_model):
    prob = _model(data)
    ann = np.argmax(prob)
    #true_target =
    #print(true_target,ann)

    #confusion_matrix[true_target][ann] += 1
    return classes[ann],prob #100*prob[0,ann]


# Visualization --------------------------------------
@st.cache_resource
def visualize_ecg(ecg,fs):
    fig = plot_ecg(uploaded_ecg=ecg, FS=fs)
    return fig


#Formatting ---------------------------------#

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {	
            visibility: hidden;
        }
        footer:after {
            content:'Made for 12-lead ECG classification with Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
        }
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('Upload ECG recording'):
    uploaded_file = st.sidebar.file_uploader(" In order for the RNN model to be able to evaluate an ECG signal, the ECG signal format must meet the following requirements: It must be a 12-lead ECG signal in microvolt, acquired with a sampling fequency of 500Hz. The signal must be uploaded as a .mat type file.", type=["mat"])

st.sidebar.markdown("")

files = {}

valfiles = []

if uploaded_file is None:
    with st.sidebar.header('Or select a ECG recording from the MIT-BIT Long-Term ECG Database for testing purposes'):
        pre_trained_ecg = st.sidebar.selectbox(
            'Select a file that was used for testing this application',
            valfiles,
            format_func=lambda x: f'{x} ({(file_gts.get(x.replace(".mat","")))})' if ".mat" in x else x,
            index=1,

        )
        if pre_trained_ecg != "None":
            f = open("data/validation/"+pre_trained_ecg, 'rb')
            if not uploaded_file:
                uploaded_file = f
        st.sidebar.markdown("Source: Physionet 2017 Cardiology Challenge")
else:
    st.sidebar.markdown("Remove the file above to demo using the validation set.")

st.sidebar.markdown("---------------")
st.sidebar.markdown("Check the [Github Repository](https://github.com/MMW1999/ECG_arrhythmia) of this project")


#---------------------------------#
# Main panel

model = get_model(f'{model_path}')

if uploaded_file is not None:
    #st.write(uploaded_file)
    

    # visualize ECG
    st.subheader('ECG visualisation')
    ecg = load_data(uploaded_file)
        
    def plot_ecg(uploaded_file, SampleRate=5000, title=filename):
      ecg_data = load_data(uploaded_file)
      ecg_plot.plot(ecg_data[0]/1000, sample_rate=SampleRate, title=title)
      ecg_plot.show()

    plot_ecg(ecg, title=uploaded_file)
        

        
    # Prediction model
    st.subheader('Model Predictions')
    with st.spinner(text="Running Model..."):
        pred,conf = get_prediction(ecg, model)
        mkd_pred_table = [
            "| Rhythm Type | Confidence |",
            "| --- | --- |"
        ]
        for i in range(len(classes)):
            mkd_pred_table.append(f"| {classes[i]} | {conf[0,i]*100:3.1f}% |")
        mkd_pred_table = "\n".join(mkd_pred_table)

        st.write("ECG classified as **{}**".format(pred))
        pred_confidence = conf[0,np.argmax(conf)]*100
        st.write("Confidence of the prediction: **{:3.1f}%**".format(pred_confidence))
        st.write(f"**Likelihoods:**")
        st.markdown(mkd_pred_table)