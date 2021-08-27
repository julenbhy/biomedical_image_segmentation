import sys
sys.path.append('../tools')
import os
import sys
import tkinter as tk
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from PIL import Image
from PIL import ImageTk
import webbrowser
import tensorflow as tf
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing

FILETYPES = [("image", ".jpeg"),
             ("image", ".jpg"),
             ]
TILE_SIZE=256
NUM_CLASSES=6

model = None
model_name = 'tiled_unet_d40_t256.hdf5'
predicts = []         # each prediction is a dictionary containing 'filename' and 'image'
input_path = None
output_path = './exports'


class Aplication():

    def __init__(cls):
        cls.window = Tk()
        cls.window.title("Image Segmentation")
        cls.window.eval('tk::PlaceWindow . center')
        cls.window.iconphoto(True, tk.PhotoImage(file='resources/logo.png'))
        # cls.window.geometry("1200x1200")
        # cls.window.columnconfigure(0, weight=1)
        # cls.window.rowconfigure(0, weight=1)
        cls.frame = None

        cls.create_menu()
        cls.window.mainloop()

    def create_menu(cls):

        menubar = Menu(cls.window)
        cls.window.config(menu=menubar)

        filemenu = Menu(menubar)
        filemenu.add_command(label="Predict single image", command=lambda: cls.single_predict_menu())
        filemenu.add_command(label="Predict from directory", command=lambda: cls.multiple_predict_menu())
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=lambda: cls.window.quit())

        helpmenu = Menu(menubar)
        helpmenu.add_command(label="Help", command=lambda: cls.help_popup())
        helpmenu.add_command(label="About...", command=lambda: cls.about_popup())

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="More", menu=helpmenu)

    def single_predict_menu(cls):

        global FILETYPES

        # Creating a Frame Container
        try:
            cls.frame.destroy()
        except:
            pass
        cls.frame = LabelFrame(cls.window, text='Make single prediction')
        cls.frame.grid(row=0, column=0, columnspan=3, pady=20)

        # Input image selection
        Label(cls.frame, text='Select input image: ').grid(row=1, column=0, padx=5, pady=5)
        Button(cls.frame, text='Select path',
               command=lambda: cls.set_file_path()) \
            .grid(row=1, column=1, padx=5, pady=5)

        # Output image
        Label(cls.frame, text='Predicted image: ').grid(row=2, column=0, padx=5, pady=5, sticky=W)

        # button Predict the result
        Button(cls.frame, text='Predict',
               command=lambda: cls.predict()) \
            .grid(row=3, columnspan=2, padx=5, pady=5)

        # button Export the result
        Button(cls.frame, text='Export',
               command=lambda: cls.export()) \
            .grid(row=4, columnspan=2, padx=5, pady=5)

    def multiple_predict_menu(cls):

        # Creating a Frame Container
        try:
            cls.frame.destroy()
        except:
            pass
        cls.frame = LabelFrame(cls.window, text='Make multiple prediction')
        cls.frame.grid(row=0, column=0, columnspan=3, pady=20)

        # Input directory
        Label(cls.frame, text='Select input directory: ').grid(row=1, column=0, padx=5, pady=5)
        Button(cls.frame, text='Select path',
               command=lambda: cls.set_dir_path(title='Select input directory', path_type='input')) \
            .grid(row=1, column=1, padx=5, pady=5)
        """
        # Output directory
        Label(cls.frame, text='Select output directory: ').grid(row=2, column=0, padx=5, pady=5)
        Button(cls.frame, text='Select path',
               command=lambda: cls.set_dir_path(title='Select output directory', path_type='output')) \
            .grid(row=2, column=1, padx=5, pady=5)
        """
        # Predict the result
        Button(cls.frame, text='Predict',
               command=lambda: cls.predict()) \
            .grid(row=3, columnspan=2, padx=5, pady=5)

        # Export the result
        Button(cls.frame, text='Export',
               command=lambda: cls.export()) \
            .grid(row=4, columnspan=2, padx=5, pady=5)

    def help_popup(cls):
        tk.messagebox.showinfo(title='Help', message='On development')

    def about_popup(cls):
        popup = Tk()
        popup.wm_title('About...')
        popup.eval('tk::PlaceWindow . center')

        Label(popup, text='Authors:'
                              '\n\tMarcos Jesus Arauzo Bravo: mararabra@yahoo.co.uk'
                              '\n\n\tJulen Bohoyo Bengoetxea: julenbhy@gmail.com')\
            .grid(row=0, column=0, sticky=W, padx=5, pady=5)

        link = tk.Label(popup, fg="blue", cursor="hand2",
              text='Github: https://github.com/julenbhy/biomedical-image-segmentation')
        link.grid(row=1, column=0, sticky=W, padx=5, pady=5)
        link.bind('<Button-1>', lambda e:  webbrowser.get('firefox')
                                            .open_new_tab(url='https://github.com/julenbhy/biomedical-image-segmentation'))

        Button(popup, text="Close", command=popup.destroy)\
            .grid(row=2, column=0, sticky=E, padx=5, pady=5)

    def set_file_path(cls):
        global input_path
        input_path = filedialog.askopenfilename(title='Select input image', filetypes=FILETYPES)

        canvas_for_image = Canvas(cls.frame, height=200, width=200,)
        canvas_for_image.grid(row=1, column=2, sticky='nesw', padx=0, pady=0)

        # create image from image location resize it to 200X200 and put in on canvas
        image = Image.open(input_path)
        canvas_for_image.image = ImageTk.PhotoImage(image.resize((200, 200), Image.ANTIALIAS))
        canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

    def set_dir_path(cls, title, path_type):
        global input_path, output_path
        path = filedialog.askdirectory(title=title)
        if(path_type == 'input'):
            input_path = path
            Label(cls.frame, text=path).grid(row=1, column=2, padx=5, pady=5)
        """
        if(path_type == 'output'):
            output_path = path
            Label(cls.frame, text=path).grid(row=2, column=2, padx=5, pady=5)
        """

    def predict(cls):
        # from tensorflow.keras.preprocessing.image import array_to_img
        global model,  TILE_SIZE, input_path, predicts
        acceptable_image_formats = [".jpg"]
        
        # if it is an only file
        if os.path.isfile(input_path):
            print('Predicting: ', input_path)
            
            # img = get_image(input_path)
            img = cv2.imread(input_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            scaler = MinMaxScaler()
            BACKBONE = 'resnet34'
            preprocess_input = sm.get_preprocessing(BACKBONE)
            img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
            
            # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
            # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
            smooth_prediction = predict_img_with_smooth_windowing(
                img,
                window_size=TILE_SIZE,
                subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                nb_classes=NUM_CLASSES,
                pred_func=(
                    lambda img_batch_subdiv: model.predict((img_batch_subdiv))
                )
            )
            prediction = np.argmax(smooth_prediction, axis=2)  #hot encoded to 1 channel
            
            import matplotlib.pyplot as plt
            f = plt.figure(figsize = (20, 20))
            f.add_subplot(1,2,1)
            plt.axis('off')
            plt. title('Original image')
            plt.imshow(img)
            f.add_subplot(1,2,2)
            plt.axis('off')
            plt. title('Prediction')
            plt.imshow(prediction)
            plt.show

            prediction = Image.fromarray((prediction * 255).astype(np.uint8))
            prediction.show()
            
            # display the prediction
            canvas_for_image = Canvas(cls.frame, height=200, width=200, )
            canvas_for_image.grid(row=2, column=2, sticky='nesw', padx=0, pady=0)
            canvas_for_image.image = ImageTk.PhotoImage(prediction.resize((200, 200), Image.ANTIALIAS))
            canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

            predicts.append({'filename': input_path.split('/')[-1], 'image': prediction})

        # if it is a directory containing multiple files
        elif os.path.isdir(input_path):
            for dir_entry in os.listdir(input_path):
                if os.path.splitext(dir_entry)[1] in acceptable_image_formats:
                    print('Predicting: ', input_path + dir_entry)
                    prediction = None
                    # prediction = model.predict(get_image(input_path+dir_entry))
                    predicts.append({'filename': dir_entry, 'image': prediction})

    def export(cls):
        """
        :param images: a list of tuples containing the path and the image (np array) to export
        """
        from tensorflow.keras.preprocessing.image import save_img
        global predicts, output_path
        cls.set_dir_path(title='Select output directory', path_type='output')
        for img in predicts:
            print('Exporting: '+output_path+img['filename'])
            # np to img
            # save_img(img['filename'], img['image'])

    def get_image(cls, path):
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scaler = MinMaxScaler()
        BACKBONE = 'resnet34'
        preprocess_input = sm.get_preprocessing(BACKBONE)
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
        return img

def main():
    global model, model_name

    print('Loading model...')
    try:
        model = tf.keras.models.load_model('../tissue_segmentation/trained_models/'+model_name, compile=False)
        #model.summary()
    except Exception:
        print (Exception)
        sys.exit('ERROR: Model '+model_name+' not found')


    mi_app = Aplication()
    return 0

if __name__ == '__main__':
    main()