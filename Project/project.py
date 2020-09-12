from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import time


def contrast_stretching(image):
    a = 0
    b = 255
    c = np.min(image)
    d = np.max(image)
    new_image = (image - c) * ((b - a) // (d - c)) + a
    return new_image.astype(np.uint8)


def normalization(image):
    img = image.copy().astype(np.float32)
    img -= np.mean(img)
    img /= np.linalg.norm(img)
    img = np.clip(img, 0, 255)
    img *= (1./float(img.max()))
    return (img*255).astype(np.uint8)


def load_image(path):
    global current_i

    if current_i > 99:
        current_path = path + str(current_i) + '.tif'
    elif current_i > 9:
        current_path = path + '0' + str(current_i) + '.tif'
    else:
        current_path = path + '00' + str(current_i) + '.tif'

    current_i += 1
    return cv2.imread(current_path, 0)


def pre_processing(image):
    image = normalization(image)
    thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=2)
    # background
    background = cv2.dilate(opening, kernel, iterations=3)
    # foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, foreground = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # finding unknown region
    foreground = np.uint8(foreground)
    unknown = cv2.subtract(background, foreground)
    _, markers = cv2.connectedComponents(foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    thresh_image = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(thresh_image, markers)
    thresh_image[markers == -1] = [0, 0, 0]
    thresh_image = cv2.cvtColor(thresh_image, cv2.COLOR_RGB2GRAY)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return thresh_image, color_image, image


def cell_tracking(image):
    global pre_center_list, total_distance, frame, line_list

    thresh_image, color_image, image = pre_processing(image)

    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_number = 0
    average_intensity = 0
    center_list = []
    dividing_count = 0

    if selected_cell != 0:
        last_cell = selected_cell_sequence[-1]
    else:
        last_cell = 0

    for index, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)  # 左上点坐标， w,h矩形宽高

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        (x1, y1), radius = cv2.minEnclosingCircle(c)
        center = (int(x1), int(y1))
        radius = int(radius)

        if w * h > 60:
            cell = {'id': index, 'center': center, 'radius': radius, 'area': w*h, 'intensity': int(image[int(y1)][int(x1)])+1, 'distance': 0, 'frame': 1, '1st_p': center}
            average_intensity += cell['intensity']
            if cell == last_cell and last_cell != 0:
                cv2.drawContours(color_image, [box], 0, (0, 0, 255))
            else:
                cv2.drawContours(color_image, [box], 0, (100, 255, 0))
            cell_number += 1
            center_list.append(cell)

    for line in line_list:
        cv2.line(color_image, line[0], line[1], (255, 255, 255))

    average_intensity = average_intensity // cell_number

    if pre_center_list:
        for c in center_list:
            min_distance = 30
            nearest_c = 0

            for cc in pre_center_list:
                temp_distance = ((c['center'][0] - cc['center'][0]) ** 2 + (c['center'][1] - cc['center'][1]) ** 2) ** 0.5
                area_ratio = min([c['area'], cc['area']]) / max([c['area'], cc['area']])
                intensity_ratio = min([c['intensity'], cc['intensity']]) / max([c['intensity'], cc['intensity']])

                # 加features-----------------------------------------------------------------------------------------------------------------#
                if temp_distance < min_distance and area_ratio > 0.5:
                    min_distance = ((c['center'][0] - cc['center'][0]) ** 2 + (c['center'][1] - cc['center'][1]) ** 2) ** 0.5
                    nearest_c = cc
                # if intensity_ratio < 0.85 or c['intensity'] > average_intensity * 2:
                #     if
            if c['intensity'] > average_intensity * 2.5:
                dividing_count += 1
                cv2.circle(color_image, c['center'], c['radius'], (255, 0, 0), 2)

            if nearest_c != 0:
                cv2.line(color_image, c['center'], nearest_c['center'], (255, 255, 255))
                line_list.append([c['center'], nearest_c['center']])
                c['distance'] = nearest_c['distance'] + min_distance
                c['frame'] = nearest_c['frame'] + 1
                c['1st_p'] = nearest_c['1st_p']

                if last_cell != 0:
                    if nearest_c == last_cell:
                        total_distance += min_distance
                        frame += 1
                        update_distance(c['distance'])
                        update_velocity(c['distance'] / c['frame'])
                        selected_cell_sequence.append(c)
                        net_distance = ((c['center'][0] - c['1st_p'][0]) ** 2 + (c['center'][1] - c['1st_p'][1]) ** 2) ** 0.5
                        update_net(net_distance)
                        if net_distance == 0:
                            update_confinement(0)
                        else:
                            update_confinement(c['distance'] / net_distance)

    pre_center_list = [_ for _ in center_list]
    cv2.putText(color_image, f'The number of cell is {cell_number}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(color_image, f'The number of dividing cell is {dividing_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return color_image


def next_image():
    image = load_image(image_path)

    new_image = cell_tracking(image)
    PIL_image = Image.fromarray(new_image)
    photo = ImageTk.PhotoImage(PIL_image)

    image_label.configure(image=photo)
    image_label.image = photo

    my_window.update_idletasks()


def play_video():
    global current_i
    global stop
    stop = 0
    while current_i <= 99 and not stop:
        image = load_image(image_path)

        new_image = cell_tracking(image)
        PIL_image = Image.fromarray(new_image)
        photo = ImageTk.PhotoImage(PIL_image)

        image_label.configure(image=photo)
        image_label.image = photo

        my_window.update_idletasks()
        time.sleep(0.2)


def stop_video():
    global stop
    stop = 0


def select_cell(event):
    global selected_cell, selected_cell_sequence, frame, total_distance
    selected_cell = 0
    selected_cell_sequence = []

    min_distance = 30
    for cell in pre_center_list:
        distance = ((cell['center'][0] - event.x) ** 2 + (cell['center'][1] - event.y) ** 2) ** 0.5
        if distance < min_distance:
            selected_cell = cell
            min_distance = distance

    if selected_cell != 0:
        frame = 0
        total_distance = 0
        update_distance(selected_cell['distance'])

        net_distance = ((selected_cell['center'][0] - selected_cell['1st_p'][0]) ** 2 + (selected_cell['center'][1] - selected_cell['1st_p'][1]) ** 2) ** 0.5
        update_net(net_distance)

        if selected_cell['frame'] == 0:
            update_velocity(0)
            update_confinement(0)
        else:
            update_velocity(selected_cell['distance'] / selected_cell['frame'])
            update_confinement(selected_cell['distance'] / net_distance)


        selected_cell_sequence.append(selected_cell)
        coordinate = selected_cell["center"]
        selected_cell_label.configure(text=f"selected cell: x={coordinate[1]}, y={coordinate[0]}")
        my_window.update_idletasks()


def update_distance(distance):
    distance_label.configure(text=f"total distance: %.2f" % distance)
    my_window.update_idletasks()


def update_velocity(velocity):
    velocity_label.configure(text=f"velocity: %.2f" % velocity)
    my_window.update_idletasks()


def update_net(net):
    net_label.configure(text=f"net distance: %.2f" % net)
    my_window.update_idletasks()


def update_confinement(confinement):
    confinement_label.configure(text=f"Confinement ratio: %.2f" % confinement)
    my_window.update_idletasks()


image_path = 'COMP9517 20T2 Group Project Image Sequences/Fluo-N2DL-HeLa/Sequence 1/t'

ini_image = cv2.imread(image_path + '000.tif', 0)
image_shape = ini_image.shape

my_window = Tk()
my_window.title("test")
my_window.geometry(f'{image_shape[1]+300}x{image_shape[0]}')

ini_PIL_image = Image.fromarray(cv2.cvtColor(ini_image, cv2.COLOR_GRAY2RGB))
ini_photo = ImageTk.PhotoImage(image=ini_PIL_image)

current_i = 0
stop = 0
selected_cell = 0
total_distance = 0
frame = 0
selected_cell_sequence = []
pre_center_list = []
line_list = []

image_label = Label(my_window, image=ini_photo)
image_label.place(x=0, y=0)
image_label.bind("<Double-Button-1>", select_cell)

selected_cell_label = Label(my_window, font=("Arial", 16), text="selected cell: none")
selected_cell_label.place(x=image_shape[1]+30, y=40)

distance_label = Label(my_window, font=("Arial", 16), text="total distance: none")
distance_label.place(x=image_shape[1]+30, y=90)

velocity_label = Label(my_window, font=("Arial", 16), text="velocity: none")
velocity_label.place(x=image_shape[1]+30, y=140)

net_label = Label(my_window, font=("Arial", 16), text="net distance: ")
net_label.place(x=image_shape[1]+30, y=190)

confinement_label = Label(my_window, font=("Arial", 16), text="Confinement ratio: ")
confinement_label.place(x=image_shape[1]+30, y=240)

Button(my_window, text='next image', command=next_image, font=("Arial", 20)).place(x=image_shape[1]+30, y=350)
Button(my_window, text='play video', command=play_video, font=("Arial", 20)).place(x=image_shape[1]+30, y=430)
Button(my_window, text='stop video', command=stop_video, font=("Arial", 20)).place(x=image_shape[1]+30, y=510)

my_window.mainloop()