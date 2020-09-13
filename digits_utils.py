from keras.preprocessing.image import ImageDataGenerator

def load_images():
    img_path='C:/Users/subhankar nath/Desktop/neural_network/Handwritten math symbol and digit dataset/train'
    
    data_gen=ImageDataGenerator(rescale=1.0/255)
    
    train_it=data_gen.flow_from_directory(directory=img_path, target_size=(64,64), batch_size=32)
    
    return train_it

def load_eval_data():
    
    image_path='C:/Users/subhankar nath/Desktop/neural_network/Handwritten math symbol and digit dataset/eval'
    
    data_gen=ImageDataGenerator(rescale=1.0/255)
    
    eval_it=data_gen.flow_from_directory(directory=image_path, target_size=(64,64))
    
    return eval_it
    
