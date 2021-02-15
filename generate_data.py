
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import time

# Material Data
Dict_M = ['Air','Ag','Alq3','C60','CuPC','SiO2','SiNx']
OriginData = np.zeros(7, dtype=object)
OriginData[1] = pd.read_csv('./Ag.dat', skipinitialspace=True)  # Ag
OriginData[2] = pd.read_csv('./Alq3.dat', skipinitialspace=True)  # Alq3
OriginData[3] = pd.read_csv('./c60.dat', skipinitialspace=True)  # c60
OriginData[4] = pd.read_csv('./cupc.dat', skipinitialspace=True)  # cupc
OriginData[5] = 1.50  # sio2
OriginData[6] = 1.91  # sinx

Points = 200+1
Lambda = np.linspace(400, 800, Points)  # case 1
j = complex(0, 1)

Index = np.zeros((7, 2, Points))
Index[0][0] = np.full(Points, 1.00)
Index[0][1] = np.zeros(Points)
for i in range(1, 5):
    Index[i][0] = np.array(splev(Lambda, splrep(OriginData[i]['lambda'], OriginData[i]['n'])))
    Index[i][1] = np.array(splev(Lambda, splrep(OriginData[i]['lambda'], OriginData[i]['k'])))
Index[5][0] = np.full(Points, 1.50)
Index[5][1] = np.zeros(Points)
Index[6][0] = np.full(Points, 1.91)
Index[6][1] = np.zeros(Points)

Ksquare = np.zeros((7, Points), dtype=complex)
for i in range(7):
    Ksquare[i] = np.ravel(np.square(np.divide(np.ravel(np.transpose(Index[i]) @ np.array([[1], [j]])), Lambda) * 2*np.pi/(10**-9)))

# Layer conditions
Set_M = np.array([5, 6])  # 0~6 : vacuum, ag, alq3, c60, cupc, sio2, sinx
set_num = len(Set_M)
REP = 4
Data_num = 50000
def layer_set(set_M=Set_M, rep=REP, data_num=Data_num):
    set_num = len(set_M)
    Set_T = np.array(np.random.choice(np.arange(40, 200), set_num*rep*data_num)).reshape(data_num, set_num*rep)
    return Set_T

def Lm(q, q_next, d):
    Q = np.array([np.array([[(q + q_next) / q_next / 2, (q_next - q) / q_next / 2], [(q_next - q) / q_next / 2, (q + q_next) / q_next / 2]]).swapaxes(0,2), ]*Data_num)
    D = np.multiply(np.eye(2),
                    np.array([np.exp(j * np.outer(q, d).ravel()), np.exp(-j * np.outer(q, d).ravel())]).swapaxes(0, 1)[
                    :, np.newaxis]).reshape(Data_num, Points, 2, 2)
    return Q @ D
vLm = np.vectorize(Lm, otypes=[np.ndarray])

def gen_dataset():
    Set_T = layer_set()
    M = np.array([np.identity(2), ]*Data_num*Points).reshape(Data_num, Points, 2, 2)
    q = np.sqrt(np.array(Ksquare[0]))
    q_next = np.sqrt(np.array(Ksquare[Set_M[0]]))
    M = np.matmul(Lm(np.ravel([q]), np.ravel([q_next]), np.ravel([0,]*Data_num)), M)
    q = q_next
    for i in range(1, set_num*REP):
        q_next = np.sqrt(np.array(Ksquare[Set_M[i % set_num]]))
        M = np.matmul(Lm(np.ravel([q]), np.ravel([q_next]), np.ravel(Set_T.transpose()[i-1]*(10**-9))), M)
        q = q_next
    q_next = np.sqrt(np.array(Ksquare[0]))
    M = np.matmul(Lm(np.ravel([q]), np.ravel([q_next]), np.ravel(Set_T.transpose()[set_num*REP-1]*(10**-9))), M)
    M = np.reshape(M, (Data_num*Points, 4)).transpose()
    r, t = -M[2]/M[3], M[0]-M[1]*M[2]/M[3]
    R, T = np.reshape(abs(r)**2, (Points, Data_num)).transpose(), np.reshape(abs(t)**2, (Points, Data_num)).transpose()
    #r, t = np.reshape(r, (Points, Data_num)).transpose(), np.reshape(t, (Points, Data_num)).transpose()
    #Ab = np.full((Data_num, Points), 1.0) - R - T
    T = np.round(T, 4)

    savexcol = ''
    for i in range(set_num*REP):
        savexcol += 'L'+str(i//set_num+1)+'_'+str(i%set_num+1) + ','
    savexcol = savexcol[:-1]
    saveycol = [",".join(wavelength) for wavelength in [Lambda.astype(str)]][0]

    np.savetxt("saveL.csv", Set_T, header=savexcol, fmt='%f', delimiter=',')
    np.savetxt("saveT.csv", T, header=saveycol,  fmt="%f", delimiter=",")

    #np.savetxt("RTtest_L.csv", Set_T, header=savexcol, fmt='%f', delimiter=',')
    #np.savetxt("RTtest_T.csv", T, header=saveycol,  fmt="%f", delimiter=",")

def gen_input():
    savexcol = ''
    for i in range(set_num*REP):
        savexcol += 'L'+str(i//set_num+1)+'_'+str(i%set_num+1) + ','
    savexcol = savexcol[:-1]
    input_set = layer_set(data_num=50000)
    np.savetxt("testLtoT_RT.csv", input_set, header=savexcol, fmt='%f', delimiter=',')

start = time.time()
gen_dataset()
#gen_input()
end = time.time()
print(np.round(end-start))