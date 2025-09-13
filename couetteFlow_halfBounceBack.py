# 1D Diffusion equation will be solved using the LBM Method
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def calc_equilibrium(rho,u,v,tw,csqr):

    eq = np.zeros((9,nx,ny))

    for ix in range(nx):
        for iy in range(ny):
            usqr = (u[0,ix,iy]**2 + u[1,ix,iy]**2)/(csqr)
            for i in range(9):
                uc = (u[0,ix,iy]*v[i,0] + u[1,ix,iy]*v[i,1])/csqr


                eq[i,ix,iy] = tw[i]*rho[ix,iy]*(1+uc+0.5*(uc**2) - 0.5*usqr)

    return eq

def calc_macro(f,v):

    rho = np.zeros((nx,ny))

    for ix in range(nx):
        for iy in range(ny):
            rho[ix,iy] = 0
            for i in range(9):
                rho[ix,iy] += f[i,ix,iy]

    u = np.zeros((2,nx,ny))
    for ix in range(nx):
        for iy in range(ny):
            for i in range(9):
                u[0,ix,iy] += v[i,0] * f[i,ix,iy]
                u[1,ix,iy] += v[i,1] * f[i,ix,iy]

            u[0,ix,iy] /= rho[ix,iy]
            u[1,ix,iy] /= rho[ix,iy]

    return rho,u


# exercising with LBM implementation
nx = 51
ny = 51
dely = 0.5

H = ny-1 - 2*dely
print(f"H={H}")

delt = 1
Re = 35
uwall = 0.1
etaR = uwall*H/Re
csqr = 1/3
maxIter = 15000
#print(etaR)

tau = 0.5 + etaR/csqr
print(f"tau = {tau}")

omega = delt/tau

fin = np.zeros((9,nx,ny))
fout = np.zeros((9,nx,ny))

rho = np.ones((nx,ny))
u = np.zeros((2,nx,ny))

for ix in range(nx):
    u[0,ix,ny-1] = uwall

#print(u[0,:,:])

# custom implementation of D2Q9.
# 0: North-east
# 1: east
# 2: south-east
# 3: north
# 4: rest
# 5: south
# 6: north-west
# 7: west
# 8: south-west
v = np.array([[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]])

# weights for D2Q9
tw = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])

opp = [8,7,6,5,4,3,2,1,0]


feq = calc_equilibrium(rho,u,v,tw,csqr)


# initialising with equilibrium values
fin = feq.copy()

for time in range(maxIter):
    print(f"iteration = {time}")

    rho,u = calc_macro(fin,v)
    for ix in range(nx):
        u[0,ix,ny-1] = uwall
        u[1,ix,ny-1] = 0.0
        u[0,ix,0] = 0.0
        u[1,ix,0] = 0.0

    feq = calc_equilibrium(rho,u,v,tw,csqr)

    # collision
    for ix in range(nx):
        for iy in range(ny):
            for i in range(9):
                #fout[i,ix,iy] = 0
                fout[i,ix,iy] = fin[i,ix,iy]*(1-omega) + feq[i,ix,iy]*omega

    #print("\nAfter collision\n")

    for ix in range(nx):
        for iy in range(1,ny-1):
            
            for i in range(9):

                xnew = ix + v[i,0]
                ynew = iy + v[i,1]
                # print(f"xnew={xnew}, ynew={ynew}")

                if xnew < 0: xnew = nx-1
                if xnew > nx-1: xnew = 0
                #if isn[xnew,ynew] == 0:
                fin[i,xnew,ynew] = fout[i,ix,iy]
    
                
    # Apply boundary condition
    #Bottom Wall
    iy = 1
    for ix in range(nx):
        for i in [0,3,6]: # [0,3,6]
            fin[i,ix,iy] = fout[opp[i],ix,iy]

    #Top Wall - Moving
    iy = ny-2
    for ix in range(nx):
        rho_wall = rho[ix,iy]
        #rho_wall = ((fout[1,ix,iy] + fout[4,ix,iy] + fout[7,ix,iy])+ 2*(fout[0,ix,iy] + fout[3,ix,iy] + fout[6,ix,iy]))/(1+uwall)
        for i in [2,5,8]: # [2,5,8]
            #print(f"i={i},vi={v[i,0]}")
            fin[i,ix,iy] = fout[opp[i],ix,iy] + ((2*tw[i]*rho_wall*v[i,0]*uwall)/csqr)
        
    #print("\nAfter Streaming\n")

    # center
    center_x = nx // 2

    
    print("u_x")
    print(u[0,center_x,1:ny-1])


#y = np.linspace(0,ny,ny-2)
y=np.arange(1,ny-1)
y_phy = y -0.5
u_theory = uwall * y_phy / (H)
fig,ax = plt.subplots(1,1)
plt.plot(y_phy/H,u[0,center_x,1:ny-1]/uwall,label="LBM", marker='o',markerfacecolor='None', linestyle='None')
plt.plot(y_phy/H,u_theory/uwall,label="Analytical Solution")
plt.legend()
plt.xlabel('y/H')
plt.ylabel('u_x/uwall')
plt.savefig("Couette_Comparison_w_Analytical_sol.png")
#plt.show()