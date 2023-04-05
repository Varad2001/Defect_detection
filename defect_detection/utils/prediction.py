import keras
import numpy as np
from defect_detection.config import SAVED_MODELS, THRESHOLD
from defect_detection.utils import helpers



def get_predictions(img, product_name):
    img = helpers.prepare_input_img(img)

    # get the model for the given product
    model = SAVED_MODELS[f"{product_name}_model"]

    preds = model.predict(img)

    return preds


def get_bbox(img, model : keras.models.Model):
    """Returns the bounding box for the defect in the given 
    image based on the given model

    Args:
        img (np.array): numpy array of the image : (size,size, no_of_channels)
        model (keras.models.Model): Model trained for the purpose

    Returns:
        Tuple[tuple, tuple] : Returns two tuples 
    """

    feature_maps = helpers.get_feature_maps(img, model)
    feature_map_shape = feature_maps.shape[0]

    # get the weights from the second last layer
    wts_layer = model.get_layer(name=model.layers[-2].name)
    wts = wts_layer.get_weights()[0]

    # multiply weights to corresponding feature maps
    resultant_img = np.zeros(shape=(feature_map_shape, feature_map_shape))

    for i in range(0, wts.shape[0]):
        weight = wts[i][0]
        feature_img = feature_maps[:,:,i]
        resultant_img += weight * feature_img

    final_feature_map = helpers.process_final_feature_map(resultant_img)

    # extract only the defect part from the image
    defect_img = final_feature_map > THRESHOLD

    # extract the column numbers which have the defect part
    x_dim = np.max(defect_img , axis=0) * np.arange(0, defect_img.shape[1])

    # extract the boundary columns
    x_0 = (x_dim[x_dim > 0]).min()
    x_1 = (x_dim[x_dim > 0]).max()

    # similarly for y coordinate
    y_dim = np.max(defect_img, axis=1) * np.arange(0, defect_img.shape[1])
    y_0 = (y_dim[y_dim > 0]).min()
    y_1 = (y_dim[y_dim > 0]).max()


    # bounding box : rectangle coordinates
    pt1 = (x_0, y_0)
    pt2 = (x_1, y_1)


    return pt1, pt2
    


