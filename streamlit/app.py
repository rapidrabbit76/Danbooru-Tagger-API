import os
import typing as T
import streamlit as st
from PIL import Image
import requests

from settings import get_settings

env = get_settings()


@st.cache()
def predict(uploaded_images, threshold: float) -> T.List[dict]:
    files = [
        (
            "images",
            (
                image.name,
                image.getvalue(),
                image.type,
            ),
        )
        for image in uploaded_images
    ]
    res = requests.post(
        os.path.join(env.BASE_URL, "predict/tag"),
        files=files,
        data={"threshold": threshold},
        headers={},
    )
    result = res.json()
    return result


def main():
    st.title("Danbooru Tagger")

    with st.sidebar:
        with st.form("predict"):
            image_files = st.file_uploader(
                label="upload images",
                type=["png", "jpg", "webp", "jpeg", "bmp"],
                accept_multiple_files=True,
            )
            print(image_files)
            th = st.slider(
                "Score threshold", min_value=0.0, max_value=1.0, step=0.1
            )
            submitted = st.form_submit_button("Submit")

    if submitted:
        res = predict(image_files, th)

        for image, tags in zip(image_files, res):
            c1, c2 = st.columns((2, 1))
            c1.image(image, width=512)
            c2.json(tags)
            st.markdown("---")


main()
