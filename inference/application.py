import sys
sys.path.append('../tools')
import os
import glob
import sys
import shutil
import tkinter as tk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import webbrowser
#import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import threading


NUM_CLASSES=6
MODEL = None

class Aplication():

    def __init__(cls):
        cls.window = Tk()
        cls.window.title("Image Segmentation")
        cls.window.eval('tk::PlaceWindow . center')
        cls.window.iconphoto(True, tk.PhotoImage(file='resources/logo.png'))
        #cls.window.geometry("400x400")
        cls.window.columnconfigure(0, weight=1)
        cls.window.rowconfigure(0, weight=1)
        cls.frame = None
        
        
        # class variable definition
        cls.INPUT_PATH = None
        cls.OUTPUT_PATH = './exports/'
        cls.PREDICTIONS = []
        cls.TILE_SIZE = IntVar(cls.window)
        cls.TILE_SIZE.set(256) # default value
        cls.DOWNSAMPLE = IntVar(cls.window)
        cls.DOWNSAMPLE.set(40) # default value
        cls.CMAP = StringVar(cls.window)
        cls.CMAP.set('viridis') # default value
        cls.PLOT_HEATMAPS = BooleanVar(cls.window)
        cls.PLOT_HEATMAPS.set(True)
        cls.PLOT_INDIVIDUAL = BooleanVar(cls.window)
        cls.PLOT_INDIVIDUAL.set(True)


        cls.create_menu()
        cls.window.mainloop()

    def create_menu(cls):

        menubar = Menu(cls.window)
        cls.window.config(menu=menubar)
        
        predict_menu = Menu(menubar)
        predict_menu.add_command(label="Predict single image", command=lambda: cls.single_predict_menu())
        predict_menu.add_command(label="Predict from directory", command=lambda: cls.multiple_predict_menu())
        predict_menu.add_separator()
        predict_menu.add_command(label="Exit", command=lambda: cls.window.quit())
        
        menubar.add_cascade(label="Predict", menu=predict_menu)
        menubar.add_command(label="Options", command=lambda: cls.set_options_menu())
        menubar.add_command(label="Help", command=lambda: cls.help_popup())
        menubar.add_command(label="About...", command=lambda: cls.about_popup())
    
    def single_predict_menu(cls):
        # Clear the frame
        try: cls.frame.destroy()
        except: pass
        cls.frame = LabelFrame(cls.window, text='Make single prediction')
        cls.frame.pack()

        # Input image selection
        Label(cls.frame, text='Input image: ').grid(row=1, column=0, padx=5, pady=5)
        Button(cls.frame, text='Select image', command=lambda: cls.set_file_path()) \
            .grid(row=1, column=1, padx=5, pady=5)

        # Output image
        Label(cls.frame, text='Tissue Prediction: ').grid(row=2, column=0, padx=5, pady=5, sticky=W)
        
        Label(cls.frame, text='Tumor Prediction: ').grid(row=3, column=0, padx=5, pady=5, sticky=W)

        # button Predict the result
        Button(cls.frame, text='Predict', command=lambda: cls.predict_single_image()) \
            .grid(row=4, columnspan=2, padx=5, pady=5)

        # button Export the result
        Button(cls.frame, text='Export', command=lambda: cls.export()) \
            .grid(row=6, columnspan=2, padx=5, pady=5)

    def multiple_predict_menu(cls):

        # Clear the frame
        try: cls.frame.destroy()
        except: pass
        cls.frame = LabelFrame(cls.window, text='Make multiple prediction')
        cls.frame.pack()

        # Input directory
        Label(cls.frame, text='Select input directory: ').grid(row=1, column=0, padx=5, pady=5)
        Button(cls.frame, text='Select path', command=lambda: cls.set_dir_path(path_type='input')) \
            .grid(row=1, column=1, padx=5, pady=5)
        
        # Output directory
        Label(cls.frame, text='Select output directory: ').grid(row=2, column=0, padx=5, pady=5)
        Button(cls.frame, text='Select path', command=lambda: cls.set_dir_path(path_type='output')) \
            .grid(row=2, column=1, padx=5, pady=5)
        Label(cls.frame, text=cls.OUTPUT_PATH).grid(row=2, column=2, padx=5, pady=5)
        
        # Predict the result
        Button(cls.frame, text='Predict', command=lambda: cls.predict_multiple_image()) \
            .grid(row=3, columnspan=2, padx=5, pady=5)

        # Export the result
        Button(cls.frame, text='Export', command=lambda: cls.export()) \
            .grid(row=5, columnspan=2, padx=5, pady=5)

    def set_options_menu(cls):
        
        # Set temporary variables to default
        def set_default():
            TILE_SIZE.set(256) # default value
            DOWNSAMPLE.set(40) # default value
            CMAP.set('viridis') # default value
            PLOT_HEATMAPS.set(True)
            PLOT_INDIVIDUAL.set(True)
            
        # Save temporary variables to class variables and load the model
        def accept():
            cls.TILE_SIZE.set(TILE_SIZE.get())
            cls.DOWNSAMPLE.set(DOWNSAMPLE.get())
            cls.CMAP.set(CMAP.get())
            cls.PLOT_HEATMAPS.set(PLOT_HEATMAPS.get())
            cls.PLOT_INDIVIDUAL.set(PLOT_INDIVIDUAL.get())
            cls.select_model()
            
        
        # Clear the frame
        try: cls.frame.destroy()
        except: pass
        cls.frame = Frame(cls.window)
        cls.frame.pack()
        
        # Declare temparary variables
        TILE_SIZE = IntVar(cls.window)
        DOWNSAMPLE = IntVar(cls.window)
        CMAP = StringVar(cls.window)
        PLOT_HEATMAPS = BooleanVar(cls.window)
        PLOT_INDIVIDUAL = BooleanVar(cls.window)
        set_default()
        
        
        ##### IMAGE LOADING OPTIONS #####
        load_frame = LabelFrame(cls.frame, text='Loading Options')
        load_frame.pack(padx=10, pady=10)
        
        resolutions = [40, 10]
        Label(load_frame, text='Resolution Downsample: ').grid(row=0, column=0, padx=5, pady=5)
        OptionMenu(load_frame, DOWNSAMPLE, 40, *resolutions).grid(row=0, column=1, padx=5, pady=5)
        
        tile_sizes = [128, 256, 512, 1024]
        Label(load_frame, text='Tile size: (pixels)').grid(row=1, column=0, padx=5, pady=5)
        OptionMenu(load_frame, TILE_SIZE, 256, *tile_sizes).grid(row=1, column=1, padx=5, pady=5)

        
        ##### PREDICTION PLOTING OPTIONS #####
        plot_frame = LabelFrame(cls.frame, text='Plotting Options')
        plot_frame.pack(padx=10, pady=10)
        
        Label(plot_frame, text='Color Map: ').grid(row=0, column=0, padx=5, pady=5)
        cmap_box = ttk.Combobox(plot_frame, width = 20, textvariable=CMAP)
        cmap_box.grid(row=0, column=1, padx=5, pady=5)
        cmap_box['values'] = plt.colormaps()
        #cmap_box.current(2)
        
        ttk.Checkbutton(plot_frame, text="Plot Confidence Heatmaps", variable=PLOT_HEATMAPS, onvalue=True, offvalue=False,)\
            .grid(row=2, column=0, padx=5, pady=5, sticky=W)
        ttk.Checkbutton(plot_frame, text="Plot Individual Prediction", variable=PLOT_INDIVIDUAL, onvalue=True, offvalue=False,)\
            .grid(row=3, column=0, padx=5, pady=5, sticky=W)

        button_frame = Frame(cls.frame)
        button_frame.pack(padx=10, pady=10)
        Button(button_frame, text='Set Default', command=lambda: set_default()).grid(row=0, column=0, padx=5, pady=5)
        Button(button_frame, text='Accept', command=lambda: accept()).grid(row=0, column=1, padx=5, pady=5)
        Button(button_frame, text='Actual Values', command=lambda: cls.print_variables()).grid(row=0, column=2, padx=5, pady=5)
        
    def select_model(cls):
        global MODEL
        
        model_name = 'tiled_unet_d'+str(cls.DOWNSAMPLE.get())+'_t'+str(cls.TILE_SIZE.get())+'.hdf5'
        path = '../tissue_segmentation/trained_models/'
        print('\nLoading model', model_name)
        try: MODEL = load_model(path+model_name, compile=False); print('Model loaded')
        except Exception: print('ERROR: Model '+model_name+' not found. Try selecting other options')

    def help_popup(cls):
        tk.messagebox.showinfo(title='Help', message='On development')

    def about_popup(cls):
        popup = Tk()
        popup.wm_title('About...')
        popup.eval('tk::PlaceWindow . center')
        
        Label(popup, text='Authors:'
                              '\n\tMarcos Jesus Arauzo Bravo: mararabra@yahoo.co.uk'
                              '\n\n\tJulen Bohoyo Bengoetxea: julen.bohoyo@estudiants.urv.cat')\
            .grid(row=0, column=0, sticky=W, padx=5, pady=5)

        link = tk.Label(popup, fg="blue", cursor="hand2",
              text='Github: https://github.com/julenbhy/biomedical_segmentation')
        link.grid(row=1, column=0, sticky=W, padx=5, pady=5)
        link.bind('<Button-1>', lambda e:  webbrowser.get('firefox')
                                            .open_new_tab(url='https://github.com/julenbhy/biomedical_segmentation'))

        Button(popup, text="Close", command=popup.destroy)\
            .grid(row=2, column=0, sticky=E, padx=5, pady=5)

    def set_file_path(cls):
        filetypes = [("image", ".jpeg"), ("image", ".jpg"),]
        cls.INPUT_PATH = filedialog.askopenfilename(title='Select input image', filetypes=filetypes)

        canvas_for_image = Canvas(cls.frame, height=300, width=300,)
        canvas_for_image.grid(row=1, column=2, sticky='nesw', padx=0, pady=0)

        # create image from image location resize it to 200X200 and put in on canvas
        image = Image.open(cls.INPUT_PATH)
        canvas_for_image.image = ImageTk.PhotoImage(image.resize((300, 300), Image.ANTIALIAS))
        canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

    def set_dir_path(cls, path_type):
        if(path_type == 'input'):
            cls.INPUT_PATH = filedialog.askdirectory(title='Select input directory')+'/'
            Label(cls.frame, text=cls.INPUT_PATH).grid(row=1, column=2, padx=5, pady=5)
        
        if(path_type == 'output'):
            cls.OUTPUT_PATH = filedialog.askdirectory(title='Select output directory')+'/'
            Label(cls.frame, text=cls.OUTPUT_PATH).grid(row=2, column=2, padx=5, pady=5)
            
    def get_prediction(cls, path):
        global MODEL
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scaler = MinMaxScaler()
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        preprocess_input = sm.get_preprocessing('resnet34')
        img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
              
        # predict the mask
        smooth_prediction = predict_img_with_smooth_windowing(
            img,
            window_size=cls.TILE_SIZE.get(),
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=NUM_CLASSES,
            pred_func=(
                lambda img_batch_subdiv: MODEL.predict((img_batch_subdiv))
            )
        )
        return smooth_prediction

    def predict_single_image(cls):

        print('Predicting: ', cls.INPUT_PATH)
        smooth_prediction = cls.get_prediction(cls.INPUT_PATH)
        prediction = np.argmax(smooth_prediction, axis=2)  #hot encoded to 1 channel
        # Get the color map by name:
        cm = plt.get_cmap('viridis', lut=6) # lut=num_classes 
        colored_prediction = cm(prediction) # Apply the colormap like a function to any array

        # display the prediction
        colored_pil = Image.fromarray((colored_prediction[:, :, :3] * 255).astype(np.uint8))#.convert('RGB')
        canvas_for_image = Canvas(cls.frame, height=300, width=300, )
        canvas_for_image.grid(row=2, column=2, sticky='nesw', padx=0, pady=0)
        canvas_for_image.image = ImageTk.PhotoImage(colored_pil.resize((300, 300), Image.ANTIALIAS))
        canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

        cls.PREDICTIONS.append({'filename': cls.INPUT_PATH.split('/')[-1], 'prediction': smooth_prediction})

    def predict_multiple_image(cls):
        import time
        
        num_images = len(glob.glob(cls.INPUT_PATH+'*.jpg'))
        print('Detected files', num_images)
        
        pb = ttk.Progressbar(cls.frame,length=140,  mode='determinate')
        pb.grid(row=4, columnspan=2, padx=5, pady=5)
        i=1   # Counter for the pregessbar 
        for dir_entry in glob.glob(cls.INPUT_PATH+'*.jpg'):
            print('Predicting: ', dir_entry)
            # Update the pregessbar 
            pb['value'] = i*100/num_images; i+=1
            cls.frame.update_idletasks()
            # Make prediction
            #smooth_prediction = cls.get_prediction(dir_entry)
            smooth_prediction = None; time.sleep(0.5)
            cls.PREDICTIONS.append({'filename': dir_entry.split('/')[-1], 'prediction': smooth_prediction})    
        Label(cls.frame, text='Prediction Finished').grid(row=4, columnspan=2, padx=5, pady=5)
                    
    def export(cls):

        for img in cls.PREDICTIONS:
            filename=img['filename'].split('.')[0]
            path = cls.OUTPUT_PATH+filename
            print('Exporting: '+path+'.jpg')
            # Check if image has been previously predicted
            
            if os.path.exists(path): shutil.rmtree(path)
            os.makedirs(path)
            
            smooth_prediction = img['prediction']
            prediction = np.argmax(smooth_prediction, axis=2)  #hot encoded to 1 channel
            # Get the color map by name:
            cm = plt.get_cmap('viridis', lut=6) # lut=num_classes 
            colored_prediction = cm(prediction) # Apply the colormap like a function to any array
            colored_pil = Image.fromarray((colored_prediction[:, :, :3] * 255).astype(np.uint8))#.convert('RGB')
            colored_pil.save(path+'/'+filename+'_MASK.jpg')
            

    def print_variables(cls):
        print('\n')
        print('Input Path', cls.INPUT_PATH)
        print('Output Path', cls.OUTPUT_PATH)
        print('Predictions', cls.PREDICTIONS.len())
        print('Resolution', cls.DOWNSAMPLE.get())
        print('Tile Size', cls.TILE_SIZE.get())
        print('Cmap', cls.CMAP.get())
        print('Plot Heatmap', cls.PLOT_HEATMAPS.get())
        print('Plot Individual', cls.PLOT_INDIVIDUAL.get())
        
def main():
    global MODEL

    default_model = 'tiled_unet_d40_t256.hdf5'
    print('Loading model', default_model)
    try: MODEL = load_model('../tissue_segmentation/trained_models/'+default_model, compile=False)
    except Exception: sys.exit('ERROR: Model '+default_model+' not found')


    mi_app = Aplication()
    return 0

if __name__ == '__main__':
    main()