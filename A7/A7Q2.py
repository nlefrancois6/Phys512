import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

"""
def pot2d_handleInf(x,y, lamb=1, soft = 0.1):
    #eps = 8.85*10**(-12)
    eps = 1
    V = np.zeros([len(x),len(y)])
    
    r = R(x,y)
    mask_r = r < soft #avoid blowup at log(0)
    for i in range(len(x)):
        for j in range(len(y)):
            if mask_r[i,j]==False:
                V[i,j] = np.log(r[i,j])*lamb/(2*np.pi*eps)
            else:
                V[i,j] = np.log(r[i,j]+soft)*lamb/(2*np.pi*eps)
    #V[mask_r == True] = np.log(r+soft)*lamb/(2*np.pi*eps)
    #V[mask_r == False] = np.log(r)*lamb/(2*np.pi*eps)
    
    return V
"""

def R(x,y):
    return np.sqrt(x**2 + y**2)
#Calculate potential but don't handle the singularity at r=0
def pot2d_naive(x,y, lamb=1):
    #eps = 8.85*10**(-12)
    eps = 1
    
    r = R(x,y)
    V = np.log(r)*lamb/(2*np.pi*eps)
    return V

def greenFunc(x, y, soft = 0.1):
    r = R(x,y)
    r[r<soft] = soft

    g = 1/(4*np.pi*r)
    return g
"""
def convolve(f, g, p=10):
    #Take the convolution of f and g
    F = np.fft.fft(np.pad(f, [0, p]))
    G = np.fft.fft(np.pad(g, [0, p])) 
    conv = np.fft.ifft(F * G)
    if p > 0:
        conv= conv[:-p,:-p]
    conv = conv.real
    return conv
"""
def conv_basic(f, g):
    F = np.fft.fft(f)
    G = np.fft.fft(g)
    return np.fft.ifft(F*G)/len(f)

def conv_padded(f, g):
    #Add padding
    padLen = len(f)
    f = np.pad(f,[0,padLen],mode='constant')
    g = np.pad(g,[0,padLen],mode='constant')
    #Take convolution of padded arrays, then remove padding and take real part
    conv = conv_basic(f,g)[:-padLen,:-padLen].real
    return conv

N = 64

xs = np.linspace(-N/2, N/2, N)
xp, yp = np.meshgrid(xs, xs)

#A)
V = pot2d_naive(xp, yp)
#Fix the singularity using the average value of points surrounding [1,0] to get the value at [0,0]
originInd = int(N/2)
V[originInd, originInd] = 4*V[originInd+1, originInd] - V[originInd+2, originInd] - V[originInd+1, originInd+1] - V[originInd+1, originInd-1]

#set rho
rho = V - 0.25*(np.roll(V,1,axis=0) + np.roll(V,-1,axis=0) + np.roll(V,1,axis=1) + np.roll(V,-1,axis=1))
#Rescale to set rho[0,0]=1
normRho = rho[originInd, originInd]
V = V/normRho
rho = V - 0.25*(np.roll(V,1,axis=0) + np.roll(V,-1,axis=0) + np.roll(V,1,axis=1) + np.roll(V,-1,axis=1))
#Shift V to set V[0,0] = 1
shift = 1 - V[originInd, originInd]
V = V + shift

"""
print('V[1,0] = ', V[1,0])
print('V[2,0] = ', V[2,0])
print('V[5,0] = ', V[5,0])

Output:
    V[1,0] =  -1.1609640474436809
    V[2,0] =  -1.0804820237218404
    V[5,0] =  -1.0218657103125848
"""

plot1 = False
if plot1==True:
    plt.figure()
    plt.title('Point Charge Potential')
    plt.pcolormesh(V, cmap = cm.plasma)
    plt.colorbar()
    plt.show()
    
#B)
def Ax(g, rho, mask): 
    g_rho = conv_padded(g, rho) 
    g_rho_copy = g_rho.copy()
    g_rho_copy = g_rho_copy*mask
    return g_rho_copy

#Set Dirichlet BC on the potential
V = np.zeros([N,N])
boxL = 10
V[originInd-boxL:originInd+boxL,originInd-boxL:originInd+boxL] = 1.0
mask = V > 0

rho = np.zeros([N,N])

#Conjugate gradient solver for rho on the mask
nIter = 1000
pr = False

green = greenFunc(xp, yp)

b = V.copy()
r = b - Ax(green, rho, mask)
p = r.copy()
rTr = np.sum(np.dot(r, r))
for i in range(nIter):
    Ap = Ax(green, p, mask)
    alpha = np.dot(r,r)/np.sum(Ap*p)
    rho = rho + alpha*p
    r_new = r - alpha*Ap
    rTr_new = np.sum(np.dot(r_new, r_new))
    beta = rTr_new/rTr
    p = r_new + beta*p
    r = r_new
    rTr = rTr_new
    if (i%100==0) and (pr):
        print('Step ', i, ' Res Squared: ', rTr)
        
"""
Output:
    Step  0  Res Squared:  36.023293633611175
    Step  100  Res Squared:  2.7188329562139963
    Step  200  Res Squared:  1.7461090516424267
    Step  300  Res Squared:  1.34212170744768
    Step  400  Res Squared:  1.1116389790049797
    Step  500  Res Squared:  0.9595647409942089
    Step  600  Res Squared:  0.8503575150290216
    Step  700  Res Squared:  0.7674410689356561
    Step  800  Res Squared:  0.7019485126263549
    Step  900  Res Squared:  0.6486684493721879
    
See A7Q2b_rhoField_plot.png and A7Q2b_rhoAlongSide_plot.png for the output plots from Part B)
"""
plot2 = False
if plot2==True:
    plt.figure()
    plt.title('Charge Density, Conjugate Gradient')
    plt.pcolormesh(rho, cmap = cm.plasma)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.title('Charge Density, Conjugate Gradient')
    plt.plot(yp[originInd-boxL:originInd+boxL], rho[originInd-boxL:originInd+boxL, originInd-boxL])
    plt.show()

#C)
#I dont think this is how I'm supposed to get V from rho but i'm not sure how
V_conjGrad = conv_padded(green,rho)

plot3 = True
if plot3==True:
    plt.figure()
    plt.title('Potential, Conjugate Gradient')
    plt.pcolormesh(green, cmap = cm.plasma)
    plt.colorbar()
    plt.show()