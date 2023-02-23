

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit_authenticator as stauth

import database as db

users = db.fetch_all_users()
usernames = [user["key"]for user in users]
names = [user["name"]for user in users]
hashed_passwords = [user["password"]for user in users]

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
   "glaucomadetect", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login","main")

if authentication_status == False:
    st.error("username/password is incorrect TRY AGAIN")

if authentication_status == None:
    st.warning("please enter your username and password")

if authentication_status == True:
    st.set_option('deprecation.showfileUploaderEncoding', False)
    def import_and_predict(image_data, model):
        image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        st.image(image, channels='RGB')
        image = (image.astype(np.float32) / 255.0)
        img_reshape = image[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

    model = tf.keras.models.load_model('my_model2.h5')
    st.write("""
         # ***Glaucoma detector***
         """
         )

st.write("This is a simple image classification web app to predict glaucoma through fundus image of eye")

file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])

if file is None:
    st.text("You haven't uploaded a jpg image file")
else:
    imageI = Image.open(file)
    prediction = import_and_predict(imageI, model)
    pred = prediction[0][0]
    if(pred > 0.5):
        st.write(
                 """
                 # **Prediction:** You eye is Healthy. Great!!
                 """)
    if(pred < 0.3):
         st.write("""
                 ## **Prediction:** You are severely affected by Glaucoma."""
                 )

        

    else:
        st.write("""
                 ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """)

