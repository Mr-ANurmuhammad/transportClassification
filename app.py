import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title 
st.title("Transportni klassifikatsiya qiluvchi model!")

# rasmni joylash
file = st.file_uploader("Rasmni yuklash", type=("png", "jpeg", "jpg","svg"))



if file is not None:
    # PIL convert
    img = PILImage.create(file)
    st.image(file)
    # model
    model = load_learner("transport_model1.pkl")

    #predict
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:1f}%")
    # Plotting
    fig = px.bar(x=probs, y=model.dls.vocab)
    st.plotly_chart(fig)
else:
    st.warning("Iltimos, rasm yuklang!")