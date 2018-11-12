@ECHO OFF
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install stop_words

SET /P answer=Do you want to start the training for the image classifier?[Y/N]
if /I "%answer%" EQU "Y" goto :yes
if /I "%answer%" EQU "N" goto :no

:yes
python scripts/retrain.py --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --image_dir=tf_files/photos
goto no

:no
SET /P answer=Do you want to run the embedding?[Y/N]
if /I "%answer%" EQU "Y" goto :yes2
if /I "%answer%" EQU "N" goto :no2


:yes2
python scripts/main.py

:no2
pause
exit
