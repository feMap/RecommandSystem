#ecommending user or question according dataset and results froms Near.py.
import numpy as np
from Map import *
from Data import *
from RBM import *
from Factorize import *









if __name__ == '__main__':
    qd = Map_load('q')
    ud = Map_load('u')
    mf = MF()
    mf.load_data(path='../map_invited_info.txt', sep='\t', format={'col':0,'row':1, 'value':2, 'ids':'int'},split=False)
    #mf.load_weight('tempweight_1000.pkl')
    mf.factorize(k=50,iter=1999,pp=True)
    mf.save_weight('weight30_iter1999.pkl')
    output_path = "./output.txt"
    output_file = open(output_path,'w')
    validate_file = file("../validate_nolabel.txt")
    
    for line in validate_file:
        line = line.strip('\r\n').split(',')
        question_id = line[0]
        user_id = line[1]
        try:
            predict = mf.predict(ud[user_id],qd[question_id])
        except:
            predict = 0
            print question_id + "," + user_id + "      Exception"

        if predict > 1.0: predict = 1.0
        if predict < 0.0001:predict = 0.0
        result = question_id + "," + user_id + "," + str(predict)
        #print result

        output_file.write(result)
        output_file.write("\n")
