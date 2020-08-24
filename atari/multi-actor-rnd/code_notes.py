
# auto encoder
"""
decode = Conv2D(4, 8, strides=4, activation="relu")(norm)
decode = Conv2D(32, 4, strides=2, activation="relu")(decode)
decode = Conv2D(32, 3, strides=2, activation="relu")(decode)
decode = Dense(256, activation="relu")(decode)
decode = Conv2DTranspose(32, 3, strides=2, activation="relu")(decode)
decode = Conv2DTranspose(32, 4, strides=2, activation="relu")(decode)
decode = Conv2DTranspose(4, 8, strides=4, activation="relu")(decode)
"""

