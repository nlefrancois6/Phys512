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

def convolve(f, g, p=10):
    #Take the convolution of f and g
    F = np.fft.fft(np.pad(f, [0, p]))
    G = np.fft.fft(np.pad(g, [0, p])) 
    conv = np.fft.ifft(F * G)
    if p > 0:
        conv= conv[:-p,:-p]
    conv = conv.real
    return conv
        
N = 9

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

print('V[1,0] = ', V[1,0])
print('V[2,0] = ', V[2,0])
print('V[5,0] = ', V[5,0])

"""
Output:
    V[1,0] =  -1.1609640474436809
    V[2,0] =  -1.0804820237218404
    V[5,0] =  -1.0218657103125848
"""

plot = True
if plot==True:
    plt.figure()
    plt.title('Point Charge Potential')
    plt.pcolormesh(V, cmap = cm.plasma)
    plt.colorbar()
    plt.show()
    
#B)
green = greenFunc(xp, yp)
V = convolve(green,rho)



