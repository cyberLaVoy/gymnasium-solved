import tensorflow.keras as keras

def createNeuralNetwork():
    model = keras.models.Seqouential()
    model.add( keras.layers.Dense( 50, input_dim=2, activation="sigmoid" ) ) # input_dim defines how many input values
    model.add( keras.layers.Dense( 50, activation="sigmoid" ) )
    model.add( keras.layers.Dense( 2, activation="softmax" ) )
    model.compile( loss="mse",
                   optimizer="adam",
                   metrics=["mae"])
    return model
