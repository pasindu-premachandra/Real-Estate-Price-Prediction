import pickle
import json
import numpy as np

__locations = None
__model = None
__le = None


def get_estimated_price(
    baths,
    beds,
    location,
    house_size=2.831365e04,
    land_size=215.642461,
    lat=79.979304,
    lon=6.908008,
    seller_type=1,
):
    ar = np.ones(8)
    ar[0] = baths
    ar[1] = land_size
    ar[2] = beds
    ar[3] = house_size
    ar[4] = __le.transform([location])
    ar[5] = lat
    ar[6] = lon
    ar[7] = seller_type
    return __model.predict([ar])[0]


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations
    global __le

    # Load the label encoder
    with open("./artifacts/label_encoder.pickle", "rb") as f:
        __le = pickle.load(f)
        __locations = list(__le.classes_)

    global __model
    if __model is None:
        with open("./artifacts/finalize_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_location_names():
    return __locations


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(
        get_estimated_price(
            location=" Matara City,  Matara", baths=2, beds=3, house_size=12000
        )
    )
    print(get_estimated_price(location=" Matara City,  Matara", baths=5, beds=6))
