import cv2 
import glob 
import os 
import numpy as np
from tabulate import tabulate
import natsort

kaskade = [["Row number","Name"],
            [1,"haarcascade_eye.xml"],
            [2,"haarcascade_eye_tree_eyeglasses.xml"],
            [3,"haarcascade_frontalcatface.xml"],
            [4,"haarcascade_frontalcatface_extended.xml"],
            [5,"haarcascade_frontalface_alt.xml"],
            [6,"haarcascade_frontalface_alt2.xml"],
            [7,"haarcascade_frontalface_default.xml"],
            [8,"haarcascade_frontalface_alt_tree.xml"],
            [9,"haarcascade_fullbody.xml"],
            [10,"haarcascade_lefteye_2splits.xml"],
            [11,"haarcascade_licence_plate_rus_16stages.xml"],
            [12,"haarcascade_lowerbody.xml"],
            [13,"haarcascade_profileface.xml"],
            [14,"haarcascade_righteye_2splits.xml"],
            [15,"haarcascade_russian_plate_number.xml"],
            [16,"haarcascade_smile.xml"],
            [17,"haarcascade_upperbody.xml"]]

kaskade_one = []
for i in range(1,18):
    kaskade_one.append(kaskade[i][1])
print(tabulate(kaskade))
print("Izberi kaskado: , napisi default za privzeta vrednost")
kaskado = input()
while True:
        if kaskado not in kaskade_one:
            if kaskado != 'default':
                print("Enter the exact cascade as above: ")
                kaskado = input()
        if (kaskado in kaskade_one):
            kaskada_classifier = cv2.CascadeClassifier(kaskado)
            break
        if kaskado == 'default':
            kaskado = 'haarcascade_frontalface_default.xml'
            kaskada_classifier = cv2.CascadeClassifier(kaskado)
            break 
    
print("Izberi scale factor(double): , napisi default za privzeta vrednost")
scale_factor = input()

 # default
if scale_factor == 'default':
    scale_factor = float(1.3)
    
print("Izberi minNeighbours(int): , napisi default za privzeta vrednost")
min_neighbours = input()
    
# default 
if min_neighbours == 'default':
    min_neighbours = int(5)
    
print("Izberi minSize x(int): , napisi default za privzeta vrednost")
min_size_x = input()

# default 
if min_size_x == 'default':
    min_size_x = int(30)

print("Izberi minSize y(int): , napisi default za privzeta vrednost")
min_size_y = input()

# default 
if min_size_y == 'default':
    min_size_y = int(30)


# function for choosing fddb text file and saving that in final text file
def fddb_file(input, out_text, out_final):
    with open(input) as f:
        lines = f.readlines()
    f.close()
    
    
    for index, value in enumerate(lines):
        i_list = list(value)
        i_list.remove('\n') # remove new line character, for the path below
        i_list_joined = "".join(i_list)
       
       
    # this code is for windows thats why is \ not /
        i_replaced = [w.replace('/', '\\') for w in i_list]
    
        i_replaced_joined = "".join(i_replaced)
    
        # you must have originalPics folder where is detection_face.py
        path = 'originalPics\\' + os.path.join(i_replaced_joined + '.jpg')
        #print(path)
        if not os.path.exists(path):
            print("The path does not exist")
        
        read_img = cv2.imread(path)
        img_gray = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY) # need to be grayscaled image

        
            
        faces, reject_levels, level_weights = kaskada_classifier.detectMultiScale3(img_gray, float(scale_factor), int(min_neighbours),  minSize = (int(min_size_x), int(min_size_y)),  outputRejectLevels = True) # return x,y,w,h if faces are found
        level_weights = np.round(level_weights, 5)
        
           
        # save the results into detections folder, change the path to test on your PC
        save_path = r'C:\Users\cr008\OneDrive\Desktop\face_detection\FDDB-folds\detections'
        complete_path = os.path.join(save_path, out_text)
        complete_path_final = os.path.join(save_path, out_final)
        with open(complete_path,'a') as out:
                out.write(str(i_list_joined) + "\n")
                out.write(str(len(faces)) + "\n")
                for b in range(0,len(faces)): # for every detected face write x,y,w,h
                    out.write((str(faces[b])[1:-1]) + " " + (str(level_weights[b])) + '\n') # remove the brackets from the list [1:-1] --> first character and last
        out.close()
        input_results = open(complete_path,'r')
        output_results = open(complete_path_final,'w') # final txt to remove any multiple whitespaces between x,y,w,h
        for line in input_results:
           output_results.write(' '.join(line.split()) + '\n')
        input_results.close()
        output_results.close()

       
        i = 0
        for  x,y,w,h in faces:
            cv2.rectangle(read_img,(x,y),(x+w,y+h),(255,0,0),2)
            confidence = str(level_weights[i])
            rec_gray = img_gray[y:y+h, x:x+w]
            rec_color = read_img[y:y+h, x:x+w]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(read_img, confidence, (x,y), font, 0.5, (0,0,255), 2)
            cv2.putText(read_img,'Number of detected faces : ' + str(len(faces)),(10, 20), font, 0.5,(0,255,255),2)
            # comment this for faster execution, if u dont go through all images, all images will be not written in the final text file
            #cv2.imshow('Press any key to next image or next detected face', read_img)
            cv2.waitKey() # press any key to detect another face or go to next image
            i = i + 1 
    
          

 # do everything for every fddb fold
# for testing uncomment these one by one for faster execution 

#fddb_file("fddb-fold-01.txt", "fddb-fold-01-out.txt", "fold-01-out_windows.txt")         
#fddb_file("fddb-fold-02.txt", "fddb-fold-02-out.txt", "fold-02-out_windows.txt")
#fddb_file("fddb-fold-03.txt", "fddb-fold-03-out.txt", "fold-03-out_windows.txt")           
#fddb_file("fddb-fold-04.txt", "fddb-fold-04-out.txt", "fold-04-out_windows.txt")
#fddb_file("fddb-fold-05.txt", "fddb-fold-05-out.txt", "fold-05-out_windows.txt")
#fddb_file("fddb-fold-06.txt", "fddb-fold-06-out.txt", "fold-06-out_windows.txt")
#fddb_file("fddb-fold-07.txt", "fddb-fold-07-out.txt", "fold-07-out_windows.txt")    
#fddb_file("fddb-fold-08.txt", "fddb-fold-08-out.txt", "fold-08-out_windows.txt")
#fddb_file("fddb-fold-09.txt", "fddb-fold-09-out.txt", "fold-09-out_windows.txt")
#fddb_file("fddb-fold-10.txt", "fddb-fold-10-out.txt", "fold-10-out_windows.txt")  


################################   mine dataset   #################################
path_moje = r'C:\Users\cr008\OneDrive\Desktop\face_detection\FDDB-folds\mine_dataset'
def moja_dataset(path):
   
    img_moje_list = []
    for filename in os.listdir(path):
        img_moje = os.path.join(path, filename)
        img_moje_list.append(img_moje)
    
    img_moje_list = natsort.natsorted(img_moje_list) # sortirane, ascending order
    for i in img_moje_list:
    
        img_moje_read = cv2.imread(i)
    
    
        img_gray_moje =  cv2.cvtColor(img_moje_read, cv2.COLOR_BGR2GRAY) 
        faces_moje, reject_levels_moje, level_weights_moje = kaskada_classifier.detectMultiScale3(img_gray_moje, float(scale_factor), int(min_neighbours),  minSize = (int(min_size_x), int(min_size_y)),  outputRejectLevels = True) # return x,y,w,h if faces are found
        level_weights_moje = np.round(level_weights_moje, 5)
        c = 0
        for  x,y,w,h in faces_moje:
            cv2.rectangle(img_moje_read,(x,y),(x+w,y+h),(255,0,0),2)
            confidence = str(level_weights_moje[c])
            rec_gray = img_gray_moje[y:y+h, x:x+w]
            rec_color = img_moje_read[y:y+h, x:x+w]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_moje_read, confidence, (x,y), font, 0.5, (0,0,255), 2)
            cv2.putText(img_moje_read,'Number of detected faces : ' + str(len(faces_moje)),(10, 20), font, 0.5,(0,255,255),2)
        # comment this for faster execution, if u dont go through all images, all images will be not written in the final text file
        
            cv2.namedWindow('Press any key to next image or next detected face', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Press any key to next image or next detected face',500,500)  # resize the window for showing the image whole   
            cv2.imshow('Press any key to next image or next detected face', img_moje_read)
            cv2.waitKey() # press any key to detect another face or go to next image
        c = c + 1 
#moja_dataset(path_moje)


### anotacije moje zbirke ####

# enak vrstni red kot slike 
list_numberFaces = [2, 2, 4, 8, 1, 6, 2, 2, 3, 6, 5, 6, 21, 6, 12, 14, 32, 4, 1, 8, 10, 1, 8, 5, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1,1]

path_mine = 'mine_dataset'
paths_list = []

f = 0
for file_name in os.listdir(path_moje):
    paths = path_mine + '/' + file_name
    paths_list.append(paths)
paths_list = natsort.natsorted(paths_list)
with open("annotations.txt",'w') as pa:
    for i in paths_list:
        pa.write(i + '\n')
        pa.write(str(list_numberFaces[f]) + '\n')
        f = f + 1

 
#txt_files = glob.glob("FDDB_folds_images/*.txt")
#print(txt_files)
#with open("FDDB_folds_images\\file_path.txt", "wb") as outfile:
    #for f in txt_files:
        #with open(f, "rb") as infile:
            #outfile.write(infile.read())

        #infile.close()        
#outfile.close()


#txt_files = glob.glob("FDDB_folds_elipse/*.txt")
#print(txt_files)
#with open("FDDB_folds_elipse\\file_elipse.txt", "wb") as outfile:
    #for f in txt_files:
        #with open(f, "rb") as infile:
            #outfile.write(infile.read())        
        #infile.close()
#outfile.close()



# create manually detections_final folder and put there all fddb-fold-%2d_windows from detections folder 
'''txt_files = glob.glob("detections_final/*.txt")
#print(txt_files)
with open("detections_final\\detections_finals.txt", "w") as outfile:
    for f in txt_files:
        with open(f, "r") as infile:
            outfile.write(infile.read())
        infile.close()
outfile.close()'''
