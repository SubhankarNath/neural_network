from keras.preprocessing.image import ImageDataGenerator


def load_images():
    
    image_path='C:/Users/subhankar nath/desktop/neural_network/flowers_classification/flower_photos'
    
    data_gen= ImageDataGenerator(rescale=1.0/255)
    
    data= data_gen.flow_from_directory(image_path, target_size=(64,64), batch_size=32)
    
    return data


