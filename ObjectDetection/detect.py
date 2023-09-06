# pylint: disable=C0103
"""Detect objects in image"""

import mindspore
from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset import vision
from utils import label_map, rev_label_map, label_color_map
from model import SSD300
from PIL import Image, ImageDraw, ImageFont

checkpoint = ''
params_dict = load_checkpoint(checkpoint)

n_classes = len(label_map)
model = SSD300(n_classes=n_classes)
load_param_into_net(model, params_dict)

model.set_train(False)

resize = vision.Resize((300, 300))
to_tensor = vision.ToTensor()
normalize = vision.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    image = normalize(to_tensor(resize(original_image)))

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, _ = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    det_boxes = det_boxes[0]

    original_dims = mindspore.Tensor(
        [original_image.width, original_image.height, original_image.width, original_image.height],
        dtype=mindspore.float32).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].asnumpy().tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.shape[0]):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].asnumpy().tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image

def main():
    """main"""
    img_path = '/VOC2007/JPEGImages/000001.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()

if __name__ == '__main__':
    main()
