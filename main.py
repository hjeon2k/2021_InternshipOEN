
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


# Material Data
Dict_M = ['Air','Ag','Alq3','C60','CuPC','SiO2','SiNx']
OriginData = np.zeros(7, dtype=object)
OriginData[1] = pd.read_csv('./Ag.dat', skipinitialspace=True)  # Ag
OriginData[2] = pd.read_csv('./Alq3.dat', skipinitialspace=True)  # Alq3
OriginData[3] = pd.read_csv('./c60.dat', skipinitialspace=True)  # c60
OriginData[4] = pd.read_csv('./cupc.dat', skipinitialspace=True)  # cupc
OriginData[5] = 1.50  # sio2
OriginData[6] = 1.91  # sinx

Points = 300+1
Lambda = np.linspace(400, 700, Points)  # Lambda
Theta = np.linspace(0, np.pi/2*44/45, int((Points-1)*3/20))
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

IndexFtn = np.zeros((7, 2), dtype=object)
for i in range(7):
    IndexFtn[i][0] = splrep(Lambda, Index[i][0])
    IndexFtn[i][1] = splrep(Lambda, Index[i][1])

Ksquare = np.zeros((7, Points), dtype=complex)
for i in range(7):
    Ksquare[i] = np.ravel(np.square(np.divide(np.ravel(np.transpose(Index[i]) @ np.array([[1], [j]])), Lambda) * 2*np.pi/(10**-9)))


# Layer Matrix & title functions
'''
def Pm(q, d):
    return np.diag([np.exp(j * q * d), np.exp(-j * q * d)])
def Im(q, q_next):
    return np.array([[(q + q_next) / q_next / 2, (q_next - q) / q_next / 2], [(q_next - q) / q_next / 2, (q + q_next) / q_next / 2]])
'''
def Lm(q, q_next, d):
    return np.array([[(q + q_next) / q_next / 2, (q_next - q) / q_next / 2], [(q_next - q) / q_next / 2, (q + q_next) / q_next / 2]]) @ np.diag([np.exp(j * q * d), np.exp(-j * q * d)])
vLm = np.vectorize(Lm, otypes=[np.ndarray])

def main_title():
    title = 'Layer : '
    title += Dict_M[Seq_M[0]] + '/ '
    for i in range(1, s-1):
        title += Dict_M[Seq_M[i]] + '(' + str(Seq_T[i]) + 'nm) / '
    title += Dict_M[Seq_M[0]]
    return title

# Layer conditions
Seq_M = np.array([0, 4, 3, 0])  # 0~6 : vacuum, ag, alq3, c60, cupc, sio2, sinx
Seq_T = np.array([0, 200, 200, 500])  # 10~300nm : total thickness < 1000nm
s = len(Seq_T)

# Specific conditions
Mesh_z = int((sum(Seq_T) + 500)/10 + 1)
Mesh_y = int(1000/10 + 1)
field_Z = np.linspace(-500, sum(Seq_T), Mesh_z)
field_Y = np.linspace(-500, 500, Mesh_y)
E = np.zeros((len(field_Y), len(field_Z)), dtype=complex)
lambda0, theta0 = 500, np.pi*0.232  # target set

# Total inc angle, lambda
def cal_total():
    M = np.array([np.identity(2), ]*len(Theta)*len(Lambda))
    ky = np.resize(np.sin(Theta), (len(Theta), 1)) @ np.transpose(np.resize(np.sqrt(Ksquare[Seq_M[0]]), (Points, 1)))
    for i in range(s-1):
        q = np.sqrt(np.array([Ksquare[Seq_M[i]], ]*len(Theta)) - np.square(ky))
        q_next = np.sqrt(np.array([Ksquare[Seq_M[i+1]], ]*len(Theta)) - np.square(ky))
        M = np.matmul(np.array(vLm(np.ravel(q), np.ravel(q_next), Seq_T[i]*(10**-9)).tolist()), M)
    M = np.reshape(M, (len(Theta)*Points, 4)).transpose()
    r, t = -M[2]/M[3], M[0]-M[1]*M[2]/M[3]
    R, T = abs(r)**2, abs(t)**2
    r, t, R, T = np.reshape(r, (len(Theta), len(Lambda))), np.reshape(t, (len(Theta), len(Lambda))), np.reshape(R, (len(Theta), len(Lambda))), np.reshape(T, (len(Theta), len(Lambda)))
    Ab = np.full((len(Theta), len(Lambda)), 1.0) - R - T
    R, T, Ab = np.round(R, 10), np.round(T, 10), np.round(Ab, 10)

    fig = plt.figure(figsize=(15, 5))
    title = main_title()
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.5)
    Lambda_nm, Theta_degree = np.meshgrid(Lambda, Theta/np.pi*180)

    #graph_R = fig.add_subplot(1, 3, 1, projection='3d')
    #surf_R = graph_R.plot_surface(Lambda_nm, Theta_degree, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    graph_R = fig.add_subplot(1, 3, 1)
    surf_R = plt.imshow(R, extent=(Lambda_nm.min(), Lambda_nm.max(), Theta_degree.max(), Theta_degree.min()),
                        interpolation='nearest', cmap=cm.inferno, aspect=3)
    fig.colorbar(surf_R, shrink=0.4, aspect=10)
    graph_R.set_xlabel('Wavelength (nm)')
    graph_R.set_ylabel('Incidence Angle (°)')
    plt.title('Reflectance')

    #graph_T = fig.add_subplot(1, 3, 2, projection='3d')
    #surf_T = graph_T.plot_surface(Lambda_nm, Theta_degree, T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    graph_T = fig.add_subplot(1, 3, 2)
    surf_T = plt.imshow(T, extent=(Lambda_nm.min(), Lambda_nm.max(), Theta_degree.max(), Theta_degree.min()),
                        interpolation='nearest', cmap=cm.inferno, aspect=3)
    fig.colorbar(surf_T, shrink=0.4, aspect=10)
    graph_T.set_xlabel('Wavelength (nm)')
    graph_T.set_ylabel('Incidence Angle (°)')
    plt.title('Transmittance')

    #graph_A = fig.add_subplot(1, 3, 3, projection='3d')
    #surf_A = graph_A.plot_surface(Lambda_nm, Theta_degree, Ab, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    graph_A = fig.add_subplot(1, 3, 3)
    surf_A = plt.imshow(Ab, extent=(Lambda_nm.min(), Lambda_nm.max(), Theta_degree.max(), Theta_degree.min()),
                        interpolation='nearest', cmap=cm.inferno, aspect=3)
    fig.colorbar(surf_A, shrink=0.4, aspect=10)
    graph_A.set_xlabel('Wavelength (nm)')
    graph_A.set_ylabel('Incidence Angle (°)')
    plt.title('Absorbance')

    plt.show()


# Specific inc angle, lambda : Draw E
def Critical_angle(m1, m2):
    pass

def Ksq(ind_M=0):
    n = splev(lambda0, IndexFtn[ind_M][0])
    k = splev(lambda0, IndexFtn[ind_M][1])
    return np.square((n + k*j) * 2*np.pi/lambda0/(10**-9))

def SetSpolE(ky, Q, A, Y=field_Y, Z=field_Z):
    Y, Z = Y*(10**-9), Z*(10**-9)
    k=0
    for z_idx in range(Mesh_z):
        z = Z[z_idx]
        if z <= sum(Seq_T[:k+1])*(10**-9):
            for y_idx in range(Mesh_y):
                y = Y[y_idx]
                E[y_idx][z_idx] = np.exp(j*ky*y) * (A[k][0]*np.exp(j*Q[k]*(z-sum(Seq_T[:k])*(10**-9))) + A[k][1]*np.exp(-j*Q[k]*(z-sum(Seq_T[:k])*(10**-9))))
        else:
            k = k+1
            if k==s : break
            for y_idx in range(Mesh_y):
                y = Y[y_idx]
                E[y_idx][z_idx] = np.exp(j*ky*y) * (A[k][0]*np.exp(j*Q[k]*(z-sum(Seq_T[:k])*(10**-9))) + A[k][1]*np.exp(-j*Q[k]*(z-sum(Seq_T[:k])*(10**-9))))
    return E


def cal_target(title_layer=''):
    M = np.zeros((s, 2, 2), dtype=complex)
    M[0] = np.identity(2)
    ky = np.sqrt(Ksq(Seq_M[0])) * np.sin(theta0)
    Q = np.zeros(s, dtype=complex)
    for i in range(s-1):
        q = np.sqrt(Ksq(Seq_M[i]) - np.square(ky))
        q_next = np.sqrt(Ksq(Seq_M[i+1]) - np.square(ky))
        M[i+1] = Lm(q, q_next, Seq_T[i]*(10**-9)) @ M[i]
        Q[i] = q
    Q[s-1] = np.sqrt(Ksq(Seq_M[s-1]) - np.square(ky))
    m = M[s-1]
    r, t = -m[1][0]/m[1][1], m[0][0]-m[0][1]*m[1][0]/m[1][1]
    R, T = abs(r)**2, abs(t)**2
    print('R : ', np.round(R, 2), '  T : ', np.round(T, 2), '  A : ', np.round(1-R-T,2))

    A = (M @ np.array([[[1], [r]], ]*s)).reshape((s, 2))
    SetSpolE(ky, Q, A, Y=field_Y, Z=field_Z)
    fig = plt.figure()
    ims = []
    delta_t = 0.0
    for i in range(50):
        im = plt.imshow((E*np.exp(-j*2*np.pi*3.0*(10**8)/lambda0/(10**-9)*delta_t)).real, animated=True, cmap=cm.coolwarm,
                        interpolation='nearest', extent=[-500, sum(Seq_T), -500, 500])
        ims_addinfo = [im]
        line = [0]*(s-1)
        for k in range(s - 1):
            x, y = [sum(Seq_T[:k + 1]), sum(Seq_T[:k + 1])], [-500, 500]
            line[k], = plt.plot(x, y, color='grey')
            ims_addinfo.append(line[k])
        ims.append(ims_addinfo)
        delta_t += 0.02 * lambda0 * (10**-9) / 3.0 / (10 ** 8)

    title = 'Wavelength(nm) : ' + str(lambda0) + '   Inc Angle(°) : ' + str(np.round(theta0*180/np.pi))
    if not title_layer:
        title += '\n' + main_title()
    else:
        title += '\n' + title_layer
    plt.title(title, pad=20.0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.84)
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
    #ani.save('result.gif')
    plt.show()


# Quarter-wave stack
def quart_wave():
    global Seq_M, Seq_T, s
    global Mesh_z, Mesh_y, field_Z,field_Y, E
    global lambda0, theta0
    lambda0, theta0 = 500, 0.0
    qSeq_M = np.array([5, 6])
    qSeq_T = np.array([lambda0/splev(lambda0, IndexFtn[m][0])/4 for m in qSeq_M])
    num_pair = 3

    Seq_M, Seq_T = np.array([0]), np.array([0])
    for i in range(num_pair):
        Seq_M = np.append(Seq_M, qSeq_M)
        Seq_T = np.append(Seq_T, qSeq_T)
    Seq_M = np.append(Seq_M, np.array([0]))
    Seq_T = np.append(Seq_T, np.array([500]))

    s = len(Seq_M)
    Mesh_z = int((sum(Seq_T) + 500) / 10 + 1)
    Mesh_y = int(1000 / 10 + 1)
    field_Z = np.linspace(-500, sum(Seq_T), Mesh_z)
    field_Y = np.linspace(-500, 500, Mesh_y)
    E = np.zeros((len(field_Y), len(field_Z)), dtype=complex)

    title_layer = 'Stack of '
    for m in qSeq_M:
        title_layer += Dict_M[m] + '&'
    title_layer = title_layer[:-1] + ' with ' + str(num_pair) + ' times'
    cal_target(title_layer=title_layer)

    M = np.array([np.identity(2), ]*Points)
    ky = np.sqrt(Ksq(Seq_M[0])) * np.sin(theta0)
    for i in range(s-1):
        q = np.sqrt(Ksquare[Seq_M[i]] - np.square(ky))
        q_next = np.sqrt(Ksquare[Seq_M[i+1]] - np.square(ky))
        M = np.matmul(np.array(vLm(np.ravel(q), np.ravel(q_next), Seq_T[i]*(10**-9)).tolist()), M)
    M = np.reshape(M, (Points, 4)).transpose()
    r, t = -M[2]/M[3], M[0]-M[1]*M[2]/M[3]
    R, T = abs(r)**2, abs(t)**2
    Ab = 1 - R - T
    R, T, Ab = np.round(R, 10), np.round(T, 10), np.round(Ab, 10)

    plt.figure(figsize=(8, 5))
    Lambda_nm = Lambda
    plt.plot(Lambda_nm, R, label='Reflectance')
    plt.plot(Lambda_nm, T, label='Transmittance')
    plt.plot(Lambda_nm, Ab, label='Absorbance')
    plt.xlabel('Wavelength (nm)')
    plt.title(title_layer)
    plt.legend()
    plt.grid()
    plt.show()


#cal_target()
cal_total()
quart_wave()
