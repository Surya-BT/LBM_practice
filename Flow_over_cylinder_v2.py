# flow over cylinder using LBM
# inlet: velocity inlet using Non-Equilibrium Bounce Back (Zou/He)
# outlet: Outflow
# periodic in y

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
maxIter = 2500
nx = 101
ny = 51
dely = 0.5
H = ny-1
print(f"H={H}")
csqr = 1/3
_1bycsqr = 1/csqr
#_1bycquad = _1bycsqr**2
delt = 1
Re = 250
u_bc = 0.1
# 2D cylinder position
cyl_pos = np.array([nx//4,ny//2])
cyl_rad = ny//9

etaR = u_bc*cyl_rad/Re
tau = 0.5 + etaR/csqr
print(f"tau = {tau}")



omega = delt/tau

fin = np.zeros((9,nx,ny))
fout = np.zeros((9,nx,ny))

rho = np.ones((nx,ny))
u = np.zeros((2,nx,ny))

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

def inivel(d, x, y):
    return (1-d) * u_bc * (1 + 1e-4*np.sin(y/H*2*np.pi))

vel = np.fromfunction(inivel, (2,nx,ny))

# book keeping for wall nodes
# 0 for fluid
# 1 for no slip wall
isn = np.zeros((nx,ny))
for ix in range(nx):
    for iy in range(ny):
        if ((ix-cyl_pos[0])**2+(iy-cyl_pos[1])**2) < cyl_rad**2:
            isn[ix,iy] = 1

fin = calc_equilibrium(rho,vel,v,tw,csqr)

###### Main time loop ##########################################################
for time in range(maxIter):

    # outflow condition
    for i in [6,7,8]:
        fin[i,-1,:] = fin[i,-2,:]

    rho,u = calc_macro(fin,v)

    # velocity inlet
    u[:,0,:] = vel[:,0,:] # maintain the dirichlet condition
    
    # calculate rho at left wall ( NEBB condition )
    for iy in range(ny):
        sum_i = 0
        sum_j = 0
        for i in [3,4,5]:
            sum_i += fin[i,0,iy]
        for j in [6,7,8]:
            sum_j += fin[j,0,iy]

        rho[0,iy] = 1/(1-u[0,0,iy])*(sum_i + 2*sum_j)

    
    # equilibrium
    feq = calc_equilibrium(rho,u,v,tw,csqr)

    # NEBB at the inlet ( non-equlibrium correction )
    for i in [0,1,2]:
        for iy in range(ny):
            fin[i,0,iy] = fin[opp[i],0,iy] + feq[i,0,iy] - feq[opp[i],0,iy]

    # Collision
    for ix in range(nx):
        for iy in range(ny):
            for i in range(9):
                #fout[i,ix,iy] = 0
                fout[i,ix,iy] = fin[i,ix,iy]*(1-omega) + feq[i,ix,iy]*omega

    # streaming step
    for ix in range(nx):
        for iy in range(ny):
            for i in range(9):

                xnew = ix + v[i,0]
                ynew = iy + v[i,1]
                # print(f"xnew={xnew}, ynew={ynew}")

                if xnew < 0: xnew = nx-1
                if xnew > nx-1: xnew = 0
                if ynew < 0: ynew = ny-1
                if ynew > ny-1: ynew = 0
    
                fin[i,xnew,ynew] = fout[i,ix,iy]

    
    # bounce back
    for ix in range(nx):
        for iy in range(ny):
            for i in range(9):
                xnew = ix + v[i,0]
                ynew = iy + v[i,1]

                if ynew < 0: ynew = ny-1
                if ynew > ny-1: ynew = 0
                if xnew < 0: continue
                if xnew > nx-1: continue

                if isn[xnew,ynew] == 1:
                    # if the particles stream into the slip wall, then apply BC.
                    #print(f"iy={iy}")
                    #print(f"i={i}, opp[i]={opp[i]}")
                    fin[opp[i],ix,iy] = fin[i,xnew,ynew]
    


    if (time%100==0):
        print("saving figure")
        plt.clf()
        plt.imshow(np.sqrt(u[0,:,:]**2+u[1,:,:]**2).transpose(), cmap=cm.Reds)
        plt.savefig("myTrail/vel.{0:04d}.png".format(time//100))
    


