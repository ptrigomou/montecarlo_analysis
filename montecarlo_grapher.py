"""
montecarlo_grapher.py (v4)
PTM -- 16.01.13
Makes graphics of Montecarlo RDC bootstrapping
usage: python montecarlo_grapher.py outname
"""
from pylab import *
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
from numpy.lib.function_base import linspace
import matplotlib.ticker as tkr
import matplotlib.axis as axs
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import colorbar
import sys

qang = loadtxt("Qang.tbl")
popo = loadtxt("pop1.tbl")
beta = loadtxt("gAng.tbl")
gdom = loadtxt("gdo.tbl")
bnum = (len(qang) / 4)
#rmsd = loadtxt("bkr_1.txt")
#tit=loadtxt("title.txt",dtype='string')

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


#count=0
output = outname + '_histograms.pdf'
pdf = PdfPages(output)
rc('axes',linewidth=2) 
figure(figsize=(10,6)).subplots_adjust(left=0.08,bottom=0.08,right=0.98,top=0.95,wspace=0.22,hspace=0.40)

subplot(2,2,1).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,1).yaxis.set_major_locator(MaxNLocator(4))
# histogram definition. Normalized ie. the integral over the hist area is 1.
hist = qang
n, bins, patches = plt.hist(hist, bnum, normed=False, facecolor='blue', alpha=0.75, histtype='stepfilled')
# centering the rectangles over the bins
bincenters = 0.5*(bins[1:]+bins[:-1])
# best fit line for the normal Probability Density Function (PDF). Normalized to the experimental PDF maximum.
mu = sp.mean(hist)
# defined as unbiased estimator of the parent distribution
sigma = sp.sqrt(sp.var(hist, ddof=1))
y = mlab.normpdf(bincenters, mu, sigma)
y = Normalize(y,n)
l = plt.plot(bincenters, y, 'r--', linewidth=2)
plt.xlabel(r'$\mathrm{Q_{ang}}$')
plt.ylabel('Frequency')
#define label
titelin ='$\mathrm{Histogram\ of\ Q_{ang}}\ \mu=$'+str(round(mu,2))+'$\ \sigma=$'+str(round(sigma,2))
plt.title(titelin)
#plt.xlim(40, 160)
#plt.ylim(0, 50)
plt.grid(True)


subplot(2,2,2).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,2).yaxis.set_major_locator(MaxNLocator(4))
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
titelin ='$\mathrm{Histogram\ of\ p_{(1)}}\ \mu=$'+str(round(mu,2))+'$\ \sigma=$'+str(round(sigma,2))
plt.title(titelin)
#plt.xlim(40, 160)
#plt.ylim(0, 50)
plt.grid(True)


subplot(2,2,3).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,3).yaxis.set_major_locator(MaxNLocator(4))
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
titelin ='$\mathrm{Histogram\ of\ Beta\ angle}\ \mu=$'+str(round(mu,2))+'$\ \sigma=$'+str(round(sigma,2))
plt.title(titelin)
#plt.xlim(40, 160)
#plt.ylim(0, 50)
plt.grid(True)


subplot(2,2,4).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,4).yaxis.set_major_locator(MaxNLocator(4))
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
titelin ='$\mathrm{Histogram\ of\ Q_{ang}}\ \mu=$'+str(round(mu,2))+'$\ \sigma=$'+str(round(sigma,2))
plt.title(titelin)
#plt.xlim(40, 160)
#plt.ylim(0, 50)
plt.grid(True)


#plt.show()
pdf.savefig()
close()
pdf.close()




output = outname + '_scatter.pdf'
pdf = PdfPages(output)
rc('axes',linewidth=2) 
figure(figsize=(10,6)).subplots_adjust(left=0.08,bottom=0.08,right=0.98,top=0.95,wspace=0.22,hspace=0.40)

subplot(2,2,1).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,1).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(popo,qang,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(popo, qang, 'bo', popo, fit_y(popo), '-r',linewidth=2)
plt.title(r'$\mathrm{p_{(1)} vs. Q_{ang}}$')
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel(r'$\mathrm{Q_{ang}}$')
#plt.xlim(0,80)
#plt.ylim(0,10)


subplot(2,2,2).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,2).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(popo,beta,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(popo, beta, 'bo', popo, fit_y(popo), '-r',linewidth=2)
plt.title(r'$\mathrm{p_{(1)} vs. \beta}$')
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel(r'$\mathrm{\beta}$')
#plt.xlim(0,80)
#plt.ylim(0,10)


subplot(2,2,3).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,3).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(popo,gdom,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(popo, gdom, 'bo', popo, fit_y(popo), '-r',linewidth=2)
plt.title(r'$\mathrm{p_{(1)} vs. GDO \cdot 10^6}$')
plt.xlabel(r'$\mathrm{p_{(1)}}$')
plt.ylabel(r'$\mathrm{GDO \cdot 10^6}$')
#plt.xlim(0,80)
#plt.ylim(0,10)


subplot(2,2,4).xaxis.set_major_locator(MaxNLocator(5))
subplot(2,2,4).yaxis.set_major_locator(MaxNLocator(4))
#linear regression
fit = np.polyfit(qang,beta,1)
fit_y = poly1d(fit)
# setting up the scatter plot
plt.plot(qang, beta, 'bo', qang, fit_y(qang), '-r',linewidth=2)
plt.title(r'$\mathrm{Q_{ang} vs. \beta}$')
plt.xlabel(r'$\mathrm{Q_{ang}}$')
plt.ylabel(r'$\mathrm{\beta}$')
#plt.xlim(0,80)
#plt.ylim(0,10)


#plt.show()
pdf.savefig()
close()
pdf.close()