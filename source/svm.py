
import sys
import os
import numpy as np
from subprocess import *
from plotFunctions import *

# predict function for frame by frame processing

def SVM_predict(folder, start_frame, end_frame, type):
    svmscale_exe = r".\windows\svm-scale.exe"
    svmpredict_exe = r".\windows\svm-predict.exe"

    assert os.path.exists(svmscale_exe), "svm-scale executable not found"
    assert os.path.exists(svmpredict_exe), "svm-predict executable not found"

    testPath = "testData" + str(type) + ".txt"

    file_name = "testData" + str(type)
    model_file = "DatasetX_" + str(type) + ".txt.model"
    scaled_test_file = file_name + ".scale"
    predict_test_file = file_name + ".predict"
    range_file = "DatasetX_" + str(type) + ".txt.range"

    cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, testPath, scaled_test_file)
    #print('Scaling testing data...')
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
    #print('Testing...')
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    #print('Output prediction: {0}'.format(predict_test_file))

    with open(predict_test_file) as f:
        first_line = f.readline().rstrip()

    if str(type) == "0":
        roc = "mouth-related"
    elif str(type) == "1":
        roc = "left-eye-related"
    elif str(type) == "2":
        roc = "right-eye-related"

    if first_line == "1":
        print('{0} : {1} microexpression detected in frame window : {2} - {3}'.format(folder, roc, start_frame, end_frame))

    f.close()

# fast predict function with pre-extracted data

def fast_SVMpredict():
    for i in range(3):
        svmscale_exe = r".\windows\svm-scale.exe"
        svmpredict_exe = r".\windows\svm-predict.exe"

        assert os.path.exists(svmscale_exe), "svm-scale executable not found"
        assert os.path.exists(svmpredict_exe), "svm-predict executable not found"

        testPath = "TestingX_" + str(i) + ".txt"

        file_name = "TestingX_" + str(i)
        model_file = "DatasetX_" + str(i) + ".txt.model"
        scaled_test_file = file_name + ".scale"
        predict_test_file = file_name + ".predict"
        range_file = "DatasetX_" + str(i) + ".txt.range"

        cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, testPath, scaled_test_file)
        # print('Scaling testing data...')
        Popen(cmd, shell=True, stdout=PIPE).communicate()

        cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
        # print('Testing...')
        Popen(cmd, shell=True, stdout=PIPE).communicate()


    merge_predict("TestingX_0.predict", "TestingX_1.predict", "TestingX_2.predict")



# function for merge prediction results and plot

def merge_predict(path1, path2, path3):
    with open(path1) as f:
        numbers1 = [int(i) for i in f]
    with open(path2) as g:
        numbers2 = [int(i) for i in g]
    with open(path3) as h:
        numbers3 = [int(i) for i in h]


    for j in range(len(numbers1)):
        if numbers1[j] == 1:
            numbers1[j] = 1
        elif numbers2[j] == 1:
            numbers1[j] = 1
        elif numbers3[j] == 1:
            numbers1[j] = 1


    confusionMatrixPlot(numbers1)


# Function to calculate Chi-distace
def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(A, B)])

    return chi


# function to build SVM model

def SVM_model(trainPath):
    filename = "trainData"
    range_file = filename + ".txt.range"
    scaled_file = filename + ".txt.scale"
    model_file = filename + ".txt.model"
    svmscale_exe = r".\windows\svm-scale.exe"
    svmtrain_exe = r".\windows\svm-train.exe"
    grid_py = r".\libsvm-3.24\tools\grid.py"
    cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, trainPath, scaled_file)
    print('Scaling training data...')
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    cmd = '{0} -svmtrain "{1}" {2}"'.format(grid_py, svmtrain_exe, scaled_file)
    print('Cross validation...')
    f = Popen(cmd, shell=True, stdout=PIPE).stdout

    line = ''
    while True:
        last_line = line
        line = f.readline()
        if not line: break
    c, g, rate = map(float, last_line.split())

    print('Best c={0}, g={1} CV rate={2}'.format(c, g, rate))
    cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(svmtrain_exe, c, g, scaled_file, model_file)
    # cmd = '{0} -c 8.0 -g 0.03125  "{1}" "{2}"'.format(svmtrain_exe,scaled_file,model_file)
    print('Training...')
    Popen(cmd, shell=True, stdout=PIPE).communicate()

    return