from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np

model = models.load_model("baseline_mariya.keras")
def predict_image(model, path_to_img):
    #print(model.summary())
    #print(path_to_img)
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asarray(img)  #normalizing the images
    print("before", data[0][0])
    data = data / 255 
    print("after", data[0][0])

    probs = model.predict(data) #check if image is compatible with the network

content = ""

img_path = "placeholder_image.png"

index = """
<|text-center|
<|{"logo.png"}|image|width=20vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{img_path}|image|width=20vw|>

<|label goes here|indicator|value=0|min=0|max=100|width=20vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        state.img_path = var_val
        predict_image(model, var_val)
    #print(var_name, var_val)

app = Gui(page=index)


if __name__ == "__main__":

    app.run(use_reloader=True)