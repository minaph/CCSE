import logging

import cv2
from detectron2.engine import DefaultPredictor
import numpy as np

from common.cmd_parser import parse_cmd_arg
from initializer.instance_initializer import InferenceInstanceInitializer


from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from urllib.parse import unquote_plus
from urllib.request import urlopen

from scripts import get_all_chars, draw

app = Flask(__name__)
cors = CORS(app)

predictor: DefaultPredictor = None
dataset_metadata = None

target_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def model_init(init: InferenceInstanceInitializer):
    # this need to set in config yaml file
    # path to the model we just trained
    # config.MODEL.WEIGHTS = join('output/debug/20210608.232202', "model_final.pth")
    # set a custom testing threshold
    # config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    global dataset_metadata, predictor
    config = init.config
    dataset_metadata = init.dataset_metadata

    predictor = DefaultPredictor(config)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    logging.info("Receive request: /upload")

    img = request.form['picture']
    img = unquote_plus(img)
    logging.info("Request image_path: {0}".format(img))


    with urlopen(img) as response:
        img = response.file.read()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    img = np.array(img)
    img = np.stack([img, img, img], axis=2)

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    dict_output = {
        "image_size": instances.image_size,
        "pred_boxes": instances.pred_boxes.tensor.numpy().tolist(),
        "scores": instances.scores.numpy().tolist(),
        "pred_classes": instances.pred_classes.numpy().tolist(),
        "pred_masks": instances.pred_masks.numpy().astype(np.uint8).tolist()
    }
    logging.info(dict_output)
    return dict_output

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

@app.route('/random', methods=['POST'])
def random():
    logging.info("Receive request: /random")

    img_count = int(request.form['count'])
    logging.info("Request image_count: {0}".format(img_count))
    all_chars = get_all_chars(target_font)
    
    random_chars = np.random.choice(all_chars, img_count)
    logging.info("Random chars: {0}".format(random_chars))

    result = []

    for char in random_chars:
        img = draw(chr(char), target_font)
        img = cv2.cvtColor(pil2cv(img), cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = cv2.boxFilter(img, -1, (3, 3), normalize=True)
        char_img = np.array(img)
        img = np.stack([char_img, char_img, char_img], axis=2)

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        dict_output = {
            "source": (char_img < 128).astype(int).tolist(),
            "image_size": instances.image_size,
            "pred_boxes": instances.pred_boxes.tensor.numpy().tolist(),
            "scores": instances.scores.numpy().tolist(),
            "pred_classes": instances.pred_classes.numpy().tolist(),
            "pred_masks": instances.pred_masks.numpy().astype(np.uint8).tolist()
        }
        logging.info(dict_output)

        result.append(dict_output)

    return jsonify(result)

if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InferenceInstanceInitializer(args.config)
    model_init(initializer)
    app.run(host='0.0.0.0', port=3000)
