import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

st.title(":blue[Malaria Classification]")

image=Image.open('images (4).jpeg')
st.image(image,width=700)


model=load_model('cnnmodel (4).h5')
classes = ['parasitized','Uninfected']

def preprocessed(image):
    img=Image.open(image)
    img=img.resize((150,150))
    img_array=np.array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)
    prediction=model.predict(img_array)
    pred=np.argmax(prediction)
    return classes[pred]

file=st.file_uploader("Please choose an image",type=["jpg","jpeg","png","webp"])

if file is not None:
    image=Image.open(file)
    st.image(image,caption="Image Uploaded",use_column_width=True)
    class_name=preprocessed(file)


    if class_name=="parasitized":
        st.write(":red[RESULT: The cell is infected with Malaria]")
    else:
        st.write(":green[RESULT: The cell is free of Malaria]")




