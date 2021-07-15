import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
from PIL import Image
from PIL import ImageTk
import webbrowser
# import tensorflow


FILETYPES = [("image", ".jpeg"),
             ("image", ".png"),
             ("image", ".jpg"),
             ]
IMG_HEIGHT = 256
IMG_WIDTH = 256

model = None
predicts = []
input_path = None
output_path = './exports'


class Aplication():

    def __init__(cls):
        cls.window = Tk()
        cls.window.title("Image Segmentation")
        cls.window.eval('tk::PlaceWindow . center')
        cls.window.iconphoto(True, tk.PhotoImage(file='resources/logo.png'))
        # cls.window.geometry("600x250")
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

        if(path_type == 'output'):
            output_path = path
            Label(cls.frame, text=path).grid(row=2, column=2, padx=5, pady=5)

    def predict(cls):
        global model, input_path, predicts
        acceptable_image_formats = [".jpg", ".jpeg", ".png", ".bmp", ".bif"]
        # if it is an only file
        if os.path.isfile(input_path):
            print('Predicting: ', input_path)
            prediction = None
            # prediction = model.predict(get_image(input_path))
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
            # image = array_to_img(img)
            # save_img(img['filename'], img['image'])

    def get_image(cls, path):
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        global IMG_HEIGHT, IMG_WIDTH

        img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 3 channels
        img = img.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

def main():
    global model
    print('Loading model...')
    # model = tf.keras.models.load_model('/')
    # model.summary()

    mi_app = Aplication()
    return 0

if __name__ == '__main__':
    #
    main()