import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

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

def conv_basic(f, g, roll=False):
    F = np.fft.fft(f)
    G = np.fft.fft(g)
    ans = np.fft.ifft(F*G)/len(f)
    if roll==True:
        l2 = int(len(f)/4)
        ans = np.roll(ans, -l2, axis=1)
    return ans

def conv_padded(f, g, roll=False):
    #Add padding
    padLen = len(f)
    f = np.pad(f,[0,padLen],mode='constant')
    g = np.pad(g,[0,padLen],mode='constant')
    #Take convolution of padded arrays, then remove padding and take real part
    h = int(padLen/1)
    conv = conv_basic(f,g,roll).real
    conv = conv[:-h,:-h]
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

plot1 = True
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

#Set Dirichlet BC for the potential on the box
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
plot2 = True
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

#C
V_conjGrad = conv_padded(green, rho, roll=True)

"""
See A7Q2c_Vfield_outsideBox_plot.png, A7Q2c_Vfield_insideBox_plot.png, A7Q2c_Vfield_xdir_plot.png, and 
A7Q2c_Vfield_ydir_plot.png for the output plots from Part C).
The potential is roughly constant horizontally (x-direction) inside the box, but there is significant
dropoff vertically (y-direction) away from the centre of the box.

The potential just outside the box goes to zero immediately in the y-direction, whereas in the x-direction
it falls off more gradually (though still displaying a significant downturn at the edge of the box in the xdir_plot.png image).
In both cases however, the field is perpendicular to the equipotential box wall
as expected for our boundary conditions.
    
"""
plot3 = True
if plot3==True:
    boxL = boxL*2
    plt.figure()
    plt.title('Potential inside box, Conjugate Gradient')
    plt.pcolormesh(V_conjGrad[originInd-boxL:originInd+boxL, originInd-boxL:originInd+boxL], cmap = cm.plasma)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.title('Potential near box in y-direction, Conjugate Gradient')
    plt.plot(yp[originInd-boxL:originInd+boxL], V_conjGrad[originInd-boxL:originInd+boxL, originInd])
    plt.show()

    plt.figure()
    plt.title('Potential near box in x-direction, Conjugate Gradient')
    plt.plot(yp[originInd-boxL:originInd+boxL], V_conjGrad[originInd, originInd-boxL:originInd+boxL])
    plt.show()