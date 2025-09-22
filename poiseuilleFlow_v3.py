# 2D flow between parallel plates using the LBM Method
# this is a pressure driven flow
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

def calc_macro(f,v,Force=[0,0]):

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
            
            u[0,ix,iy] += Force[0]
            u[1,ix,iy] += Force[1]

            u[0,ix,iy] /= rho[ix,iy]
            u[1,ix,iy] /= rho[ix,iy]

    return rho,u


# exercising with LBM implementation
nx = 51
ny = 51
dely = 0.5

H = ny-1 - 2*dely
L = nx - 2
print(f"H={H}")

dpdx = 1e-5
delt = 1
#Re = 35
#ucenter = 0.1
#etaR = ucenter*H/Re
csqr = 1/3
_1bycsqr = 1/csqr
_1bycquad = _1bycsqr**2
maxIter = 15000
#print(etaR)

#tau = 0.5 + etaR/csqr
tau = 0.8
etaR = csqr*(tau - 0.5)
ucenter = (dpdx*H**2)/(8*etaR)
print(f"tau = {tau}\n etaR = {etaR}\n ucenter = {ucenter}")

omega = delt/tau

fin = np.zeros((9,nx,ny))
fout = np.zeros((9,nx,ny))

rho = np.ones((nx,ny))
u = np.zeros((2,nx,ny))


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

# book keeping for wall nodes
# 0 for fluid
# 1 for no slip wall
isn = np.zeros((nx,ny))
for ix in range(nx):
    for iy in range(ny):
        if iy == 0 or iy == ny-1:
            isn[ix,iy] = 1

print(isn)

feq = calc_equilibrium(rho,u,v,tw,csqr)


# initialising with equilibrium values
fin = feq.copy()

y=np.arange(1,ny-1)
y_phy = y - 0.5
u_theory = 0.5*dpdx*y_phy*(H-y_phy)/(etaR)

for time in range(maxIter):
    print(f"iteration = {time}")

    # calculate macro variables
    rho,u = calc_macro(fin,v,[dpdx/2,0])

    # calculate equilibrium values from the macro variables
    feq = calc_equilibrium(rho,u,v,tw,csqr)

    # collision
    for ix in range(nx):
        for iy in range(ny):
            for i in range(9):
                #fout[i,ix,iy] = 0
                source = (1 - 0.5*omega) * tw[i] * (_1bycsqr*(v[i,0] - u[0,ix,iy]) + _1bycquad*((v[i,0]*u[0,ix,iy] + v[i,1]*u[1,ix,iy])*v[i,0])) * dpdx
                fout[i,ix,iy] = fin[i,ix,iy]*(1-omega) + feq[i,ix,iy]*omega + source

    # streaming
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
    #My understanding
    for ix in range(nx):
        for iy in range(1,ny-1):
            for i in range(9):
                xnew = ix + v[i,0]
                ynew = iy + v[i,1]

                if xnew < 0: xnew = nx-1
                if xnew > nx-1: xnew = 0

                if isn[xnew,ynew] == 1:
                    # if the particles stream into the slip wall, then apply BC.
                    #print(f"iy={iy}")
                    #print(f"i={i}, opp[i]={opp[i]}")
                    fin[opp[i],ix,iy] = fin[i,xnew,ynew]

    center_x = nx // 2
    
    if (time%200==0):
        print("u_x")
        print(u[0,center_x,1:ny-1])
        print("saving figure")
        plt.clf()
        fig,axs = plt.subplots(1,2,figsize=(8,4),gridspec_kw={'width_ratios':[2,1]})
        axs[0].imshow(np.sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.jet,aspect="auto")
        #plt.subplot(1,2,2)
        axs[1].plot(u[0,center_x,1:ny-1]/max(u[0,center_x,1:ny-1]),y_phy/H,label="LBM", marker='o',markerfacecolor='None', linestyle='None')
        axs[1].plot(u_theory/max(u_theory),y_phy/H,label="Analytical Solution")
        axs[1].legend()
        axs[1].set_xlabel('u_x / U')
        axs[1].set_ylabel('y / H')
        plt.savefig("vel.{0:04d}.png".format(time//200))

y=np.arange(1,ny-1)
y_phy = y - 0.5
u_theory = 0.5*dpdx*y_phy*(H-y_phy)/(etaR)
fig,ax = plt.subplots(1,1)
plt.plot(y_phy/H,u[0,center_x,1:ny-1]/max(u[0,center_x,1:ny-1]),label="LBM", marker='o',markerfacecolor='None', linestyle='None')
plt.plot(y_phy/H,u_theory/max(u_theory),label="Analytical Solution")
plt.legend()
plt.xlabel('y / H')
plt.ylabel('u_x / U')
plt.savefig("Poiseuille_Comparison_w_Analytical_sol_test.png")

print(max(u[0,center_x,1:ny-1]))
print(max(u_theory))

