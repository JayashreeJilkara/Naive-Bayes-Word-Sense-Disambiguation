import sys
import pandas as pd
from collections import Counter
import nltk
import numpy as np
from bs4 import BeautifulSoup
import math
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
r = list(',.;():''')

def dataPreprocessing(Path):

    with open(Path,'r') as text:
        tags = BeautifulSoup(text,'html.parser')
        instance_count = len(tags.find_all('instance'))
        fold_count = math.ceil((len(tags.find_all('instance'))) / 5)
        sense_id = []
        prev = []
        next = []
        instance_id = []
        label=""
       # label = sense_id[0].split('%')


        for i in tags.find_all('instance'):
            sense_id.append(i.answer['senseid'])
            instance_id.append(i['id'])
            #print(i['id'])
            try:
                ptext = i.context.find('head').previousSibling
                ntext = i.context.find('head').nextSibling

                words = ptext.split()
                words = [w for w in words if not w in stop_words]
                words = [w for w in words if not w in r]

                words1 = ntext.split()
                words1 = [w for w in words1 if not w in stop_words]
                words1 = [w for w in words1 if not w in r]

                prev.append(words[-1])
                next.append(words1[0])
                #print(next)
            except:
                prev.append(".")
                next.append(".")

        l = sense_id[0].split("%")
        #print(l[0])
        for i in l[0]:
            label +=i
        label +=".wsd.out"
        #print(label)

    return prev,next,sense_id, fold_count,instance_count,instance_id,label

'''print(len(prev))
print(len(next))
print(len(sense_id))
print(label)'''

def train_test_senseid(start,end):
    train_senseid = sense_id[:]
    test_senseid = []
    train_feature1 = prev[:]
    test_feature1 = []
    train_feature2 = next[:]
    test_feature2 = []
    train_id = instance_id[:]
    test_id = []
    for i in range(start,end):
        test_senseid.append(sense_id[i])
        test_feature1.append(prev[i])
        test_feature2.append(next[i])
        test_id.append(instance_id[i])

    for i in test_senseid:
        if i in train_senseid:
            train_senseid.remove(i)

    for i in test_feature1:
        if i in train_feature1:
            train_feature1.remove(i)

    for i in test_feature2:
        if i in train_feature2:
            train_feature2.remove(i)

    for i in test_id:
        if i in train_id:
            train_id.remove(i)

    return train_senseid,test_senseid,train_feature1,test_feature1,train_feature2,test_feature2, test_id, train_id

#print(len(sense_id))
#print(len((train_senseid)))
#print(len(test_senseid))

def features(s1, s2, test_feature,train_feature,sense):
    s1 = {}
    s2 = {}
    for i in range(0,len(test_feature)):
        if test_feature[i] in train_feature:
            for j in range(0,len(train_feature)):
                if(train_feature[j]==test_feature[i]):
                    if(sense_id[j]==sense[0]):
                        if test_feature[i] not in s1:
                            s1[test_feature[i]]= 1
                        else:
                            s1[test_feature[i]]+=1
                    else:
                        if test_feature[i] not in s2:
                            s2[test_feature[i]] = 1
                        else:
                            s2[test_feature[i]] += 1
        else:
            if test_feature[i] not in s1:
                s1[test_feature[i]] = 0

            if test_feature[i] not in s2:
                s2[test_feature[i]] = 0
    return s1,s2

def naive_Bayes(train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2):

    c = (Counter(sense_id))
    sense = (list(c.keys()))

    s1 = {}
    s2 = {}
    features(s1,s2,test_feature1,train_feature1,sense)
    features(s1,s2,test_feature2,train_feature2,sense)

    sense_prob ={}
    for k,v in c.items():
        sense_prob[k] = v/instance_count

    l_feature_prob1 ={}
    l_feature_prob2 ={}

    for i in range(0,len(test_feature1)):
        item = test_feature1[i]
        try:
            l_feature_prob1[item] = (s1[test_feature1[i]] + 1) / (c[k] + len(s1))
        except:
            l_feature_prob1[item] = 1 / (c[k] + len(s1))

        item = test_feature1[i]
        try:
            l_feature_prob2[item] = (s2[test_feature1[i]] + 1) / (c[k] + len(s2))
        except:
            l_feature_prob2[item] = 1 / (c[k] + len(s2))

    r_feature_prob1 ={}
    r_feature_prob2 ={}
    for i in range(0,len(test_feature2)):
        item = test_feature2[i]
        try:
            r_feature_prob1[item] = (s1[test_feature2[i]] + 1) / (c[k] + len(s1))
        except:
            r_feature_prob1[item] = 1 / (c[k] + len(s1))

        item = test_feature2[i]
        try:
            r_feature_prob2[item] = (s2[test_feature2[i]] + 1) / (c[k] + len(s2))
        except:
            r_feature_prob2[item] = 1 / (c[k] + len(s2))


    total_prob1 ={}
    total_prob2 ={}
    for i in range(0,len(test_feature1)):
        x = l_feature_prob1[test_feature1[i]]
        y = r_feature_prob1[test_feature2[i]]
        z = sense_prob[sense[0]]
        total_prob1[test_feature1[i]] = ((x*y)*z)

        x = l_feature_prob1[test_feature1[i]]
        y = r_feature_prob1[test_feature2[i]]
        z = sense_prob[sense[1]]
        total_prob2[test_feature1[i]] = ((x * y) * z)

    pred_senseid = []
    for i in test_feature1:
        a = total_prob1[i]
        b = total_prob2[i]
        if(a>b):
            pred_senseid.append(sense[0])
        else:
            pred_senseid.append(sense[1])

    return pred_senseid


def accuracy(pred_senseid,test_senseid):
    r =0
    w =0
    for i in range(len(pred_senseid)):
        a = pred_senseid[i]
        b = test_senseid[i]
        if(a==b):
            r = r + 1
        else:
            w = w + 1
        res = (r/(r+w))*100
    return res




def fold1():
    train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2,test_id, train_id = train_test_senseid(0,fold_count)  # fold1
    pred_senseid = naive_Bayes(train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2,)
    acc = accuracy(pred_senseid,test_senseid)
    overall_acc.append(acc)
    print('Accuracy for Fold1:',acc,'%')
    df1 = pd.DataFrame()
    df1['Instance_ID'] = test_id
    df1['PREDICTED_SENSE_ID'] = test_senseid

    #result(test_id,pred_senseid)
    file1.write("Fold 1")
    file1.write("\n")
    file1.write('*************************************************************\n')
    file1.write("\n")
    file1.write(str(df1))



def fold2():
    train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2,test_id, train_id = train_test_senseid(fold_count, fold_count * 2)  # fold2
    pred_senseid = naive_Bayes(train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2)
    acc = accuracy(pred_senseid,test_senseid)
    overall_acc.append(acc)
    print('Accuracy for Fold2:', acc, '%')
    df2 = pd.DataFrame()
    df2['Instance_ID'] = test_id
    df2['PREDICTED_SENSE_ID'] = test_senseid

   # result(test_id,pred_senseid)
    file1.write("\n")
    file1.write("Fold 2")
    file1.write("\n")
    file1.write('*************************************************************\n')
    file1.write("\n")
    file1.write(str(df2))

def fold3():
    train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2,test_id, train_id = train_test_senseid(fold_count * 2, fold_count * 3)  # fold3
    pred_senseid = naive_Bayes(train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2)
    acc = accuracy(pred_senseid, test_senseid)
    overall_acc.append(acc)
    print('Accuracy for Fold3:', acc, '%')
    df3 = pd.DataFrame()
    df3['Instance_ID'] = test_id
    df3['PREDICTED_SENSE_ID'] = test_senseid
    file1.write("\n")
    file1.write("Fold 3")
    file1.write("\n")
    file1.write('*************************************************************\n')
    file1.write("\n")
    file1.write(str(df3))


def fold4():
    train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2,test_id, train_id = train_test_senseid(fold_count * 3, fold_count * 4)  # fold4
    pred_senseid = naive_Bayes(train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2)
    acc = accuracy(pred_senseid, test_senseid)
    overall_acc.append(acc)
    print('Accuracy for Fold4:', acc, '%')
    df4 = pd.DataFrame()
    df4['Instance_ID'] = test_id
    df4['PREDICTED_SENSE_ID'] = test_senseid
    file1.write("\n")
    file1.write("Fold 4")
    file1.write("\n")
    file1.write('*************************************************************\n')
    file1.write("\n")
    file1.write(str(df4))


def fold5():

    train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2,test_id, train_id = train_test_senseid(fold_count * 4, instance_count)  # fold5
    pred_senseid = naive_Bayes(train_senseid, test_senseid, train_feature1, test_feature1, train_feature2, test_feature2)
    acc = accuracy(pred_senseid, test_senseid)
    overall_acc.append(acc)
    print('Accuracy for Fold5:', acc, '%')
    df5 = pd.DataFrame()
    df5['Instance_ID'] = test_id
    df5['PREDICTED_SENSE_ID'] = test_senseid
    file1.write("\n")
    file1.write("Fold 5")
    file1.write("\n")
    file1.write('*************************************************************\n')
    file1.write(str(df5))


#prev,next,sense_id,fold_count,instance_count,instance_id,label = dataPreprocessing('tank.wsd')
prev,next,sense_id,fold_count,instance_count,instance_id,label = dataPreprocessing(sys.argv[1])
#print(sys.argv)
overall_acc = []
file1 = open(label, "a")
fold1()
fold2()
fold3()
fold4()
fold5()
file1.close()
print('Overall Accuracies of the folds combined:',sum(overall_acc)//5,'%')


















