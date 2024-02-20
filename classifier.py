from taipy.gui import Gui

img_path = "logo.png"

index = """
<|{img_path}|image|>
"""

app = Gui(page=index)


if __name__ == "__main__":

    app.run(use_reloader=True)