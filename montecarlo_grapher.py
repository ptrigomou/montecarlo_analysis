"""
montecarlo_grapher.py (singleOut branch)
PTM -- 16.01.13
Makes graphics of Montecarlo RDC bootstrapping
usage: python montecarlo_grapher.py outname /InputPath/ /OutPath/
"""
from pylab import *
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
from numpy.lib.function_base import linspace
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import colorbar
import sys


#array normalization to the normalized experimental PDF
def Normalize(y,n):
    minY = np.min(y)
    maxN = np.max(n)
    maxY = np.max(y)
    diff = (maxY - minY)
    y = ((y - minY) / diff)
    y = maxN*y
    return y


if sys.argv[1]:
    outname = str(sys.argv[1])
else :
    exit()

if sys.argv[2]:
    ipath = str(sys.argv[2])
else :
    exit()

if sys.argv[3]:
    opath = str(sys.argv[3])


qang = loadtxt(ipath+"Qang.tbl")
popo = loadtxt(ipath+"pop1.tbl")
beta = loadtxt(ipath+"gAng.tbl")
gdom = loadtxt(ipath+"gdo.tbl")
# defining the number of bins. Can be changed manually to a fixed number
#bnum = (len(qang) / 4)
bnum = 25
#rmsd = loadtxt("bkr_1.txt")
#tit=loadtxt("title.txt",dtype='string')



#count=0
output = opath + outname + '_statgraphs.pdf'
pdf = PdfPages(output)
rc('axes',linewidth=2) 
figure(figsize=(10,12)).subplots_adjust(left=0.08,bottom=0.08,right=0.96,top=0.95,wspace=0.22,hspace=0.40)

subplot(4,2,1).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,1).yaxis.set_major_locator(MaxNLocator(4))
# histogram definition. Normalized ie. the integral over the hist area is 1.
hist = qang
n, bins, patches = plt.hist(hist, bnum, normed=False, facecolor='blue', alpha=0.75, histtype='stepfilled')
# centering the rectangles over the bins
bincenters = 0.5*(bins[1:]+bins[:-1])
# best fit line for the normal Probability Density Function (PDF). Normalized to the experimental PDF maximum.
mu = sp.mean(hist)
# defined as unbiased estimator of the parent distribution. Now biased.
sigma = sp.sqrt(sp.var(hist, ddof=0))
y = mlab.normpdf(bincenters, mu, sigma)
y = Normalize(y,n)
l = plt.plot(bincenters, y, 'r--', linewidth=2)
plt.xlabel(r'$\mathrm{Q_{ang}}$')
plt.ylabel('Frequency')
#define label
nelem = len(hist)
titelin ='$N=$'+str(nelem)+'$\ \mu=$'+str(round(mu,3))+'$\ \sigma=$'+str(round(sigma,3))
plt.title(titelin, fontsize=10)
plt.xlim(0.0, 1.0)
plt.ylim(0, 50)
plt.grid(True)


subplot(4,2,2).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,2).yaxis.set_major_locator(MaxNLocator(4))
# histogram definition. Normalized ie. the integral over the hist area is 1.
hist = popo
n, bins, patches = plt.hist(hist, bnum, normed=False, facecolor='blue', alpha=0.75, histtype='stepfilled')
# centering the rectangles over the bins
bincenters = 0.5*(bins[1:]+bins[:-1])
# best fit line for the normal Probability Density Function (PDF). Normalized to the experimental PDF.
mu = sp.mean(hist)
# defined as unbiased estimator of the parent distribution
sigma = sp.sqrt(sp.var(hist, ddof=1))
y = mlab.normpdf(bincenters, mu, sigma)
y = Normalize(y,n)
l = plt.plot(bincenters, y, 'r--', linewidth=2)
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel('Frequency')
#define label
nelem = len(hist)
titelin ='$N=$'+str(nelem)+'$\ \mu=$'+str(round(mu,2))+'$\ \sigma=$'+str(round(sigma,2))
plt.title(titelin, fontsize=10)
plt.xlim(0.0, 1.0)
plt.ylim(0, 50)
plt.grid(True)


subplot(4,2,3).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,3).yaxis.set_major_locator(MaxNLocator(4))
# histogram definition
hist = beta
n, bins, patches = plt.hist(hist, bnum, normed=False, facecolor='blue', alpha=0.75, histtype='stepfilled')
# centering the rectangles over the bins
bincenters = 0.5*(bins[1:]+bins[:-1])
# best fit line for the normal Probability Density Function (PDF). Normalized to the experimental PDF.
mu = sp.mean(hist)
# defined as unbiased estimator of the parent distribution
sigma = sp.sqrt(sp.var(hist, ddof=1))
y = mlab.normpdf(bincenters, mu, sigma)
y = Normalize(y,n)
l = plt.plot(bincenters, y, 'r--', linewidth=2)
plt.xlabel(r'$\mathrm{\beta}$')
plt.ylabel('Frequency')
#define label
nelem = len(hist)
titelin ='$N=$'+str(nelem)+'$\ \mu=$'+str(int(mu))+'$\ \sigma=$'+str(int(sigma))
plt.title(titelin, fontsize=10)
plt.xlim(-10, 160)
plt.ylim(0, 50)
plt.grid(True)


subplot(4,2,4).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,4).yaxis.set_major_locator(MaxNLocator(4))
# histogram definition
hist = gdom
n, bins, patches = plt.hist(hist, bnum, normed=False, facecolor='blue', alpha=0.75, histtype='stepfilled')
# centering the rectangles over the bins
bincenters = 0.5*(bins[1:]+bins[:-1])
# best fit line for the normal Probability Density Function (PDF). Normalized to the experimental PDF.
mu = sp.mean(hist)
# defined as unbiased estimator of the parent distribution
sigma = sp.sqrt(sp.var(hist, ddof=1))
y = mlab.normpdf(bincenters, mu, sigma)
y = Normalize(y,n)
l = plt.plot(bincenters, y, 'r--', linewidth=2)
plt.xlabel(r'$\mathrm{GDO \cdot 10^6}$')
plt.ylabel('Frequency')
#define label
nelem = len(hist)
titelin ='$N=$'+str(nelem)+'$\ \mu=$'+str(int(mu))+'$\ \sigma=$'+str(int(sigma))
plt.title(titelin, fontsize=10)
plt.xlim(1000, 6000)
plt.ylim(0, 50)
plt.grid(True)




subplot(4,2,5).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,5).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(popo,qang,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(popo, qang, 'bo', popo, fit_y(popo), '-r',linewidth=2)
#plt.title(r'$\mathrm{p_{(1)} vs. Q_{ang}}$')
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel(r'$\mathrm{Q_{ang}}$')
# define the graph window
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)


subplot(4,2,6).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,6).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(popo,beta,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(popo, beta, 'bo', popo, fit_y(popo), '-r',linewidth=2)
#plt.title(r'$\mathrm{p_{(1)} vs. \beta}$')
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel(r'$\mathrm{\beta}$')
plt.xlim(0.0,1.0)
plt.ylim(-10,160)


subplot(4,2,7).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,7).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(popo,gdom,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(popo, gdom, 'bo', popo, fit_y(popo), '-r',linewidth=2)
#plt.title(r'$\mathrm{p_{(1)} vs. GDO \cdot 10^6}$')
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel(r'$\mathrm{GDO \cdot 10^6}$')
plt.xlim(0.0,1.0)
plt.ylim(1000,6000)


subplot(4,2,8).xaxis.set_major_locator(MaxNLocator(5))
subplot(4,2,8).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(qang,beta,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(qang, beta, 'bo', qang, fit_y(qang), '-r',linewidth=2)
#plt.title(r'$\mathrm{Q_{ang} vs. \beta}$')
plt.xlabel(r'$\mathrm{Q_{ang}}$')
plt.ylabel(r'$\mathrm{\beta}$')
plt.xlim(0.0,1.0)
plt.ylim(-10,160)


#plt.show()
pdf.savefig()
close()
pdf.close()
