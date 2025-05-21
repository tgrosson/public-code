# routines to assist in the reduction of ODGW test data

from astropy.io import fits
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy import stats, signal
from typing import Optional
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.stats import norm

colPeriod = 1/100000.                    # column clock period
rowPeriod = colPeriod * (2048.+64) + 23e-6    # row clock including init + overclock time + ODGW

colFreq = 1/colPeriod
rowFreq = 1/rowPeriod

ffPeriod = rowPeriod*2048

nRef = 4     # number of reference pixels on each edge

dummy = 1e30 # a big number to use to weed out nan

gain = 3.3                 # e- / ADU... guess
ffFluxConv = 1./ffPeriod   # from flux/read -> flux/s


# --- Helper routines --------------------------------------------------------

def writeFits( d, fileName, overwrite=False, extraHdr=None ):
    '''
    Write numPy array to specified FITS file
    '''
    
    print("Writing FITS file",fileName,"...")
    hdu = fits.PrimaryHDU(d)
    if extraHdr is not None:
        for key in extraHdr:
            if ('NAXIS' not in key) and ('BITPIX' not in key):
                # ignore certain keywords
                hdu.header[key] = extraHdr[key]
    hdul = fits.HDUList([hdu])
    hdul.writeto( fileName, overwrite=overwrite)

def loadFitsAndHdr( fileName ):
    '''
    Load FITS file into NumPy array, and also return header
    '''

    # print("Reading FITS file",fileName,"...")
    with fits.open(fileName) as f:
        data = f[0].data
        hdr = f[0].header

    return data, hdr

def loadFits( fileName ):
    '''
    Load FITS file into NumPy array
    '''

    d,h  = loadFitsAndHdr( fileName )

    return d

def calcMaskThresh( d, badRead=1, badMaxThresh=3000., badMinThresh=200. ):
    '''
    Given stack of reads, calculate bad pixel mask using thresholds

    Calculates difference between badRead and read 0, and establishes
    mask with thresholds.

    d            = cube of full-frame reads
    badRead      = read to use for mask calculation
    badMaxThresh = values greater than this are bad
    badMinThresh = values less than this are bad

    Return: mask in which good pixels are 1, and bad nan
    '''
    
    print("Calculating pixel mask using difference of reads", badRead,
          "and 0. minThresh=", badMinThresh, "maxThresh=", badMaxThresh)

    diff = d[badRead,:,:] - d[0,:,:]

    pixMask = d[0,:,:]*0. + 1.

    badIndex = np.where(diff > badMaxThresh)
    pixMask[badIndex[0], badIndex[1]] = np.nan
    badIndex = np.where(diff < badMinThresh)
    pixMask[badIndex[0], badIndex[1]] = np.nan
    
    return pixMask


def calcFluxModel(y_in, firstRead=None, lastRead=None, nSigma=3, maxIter=5,
                  smoothOutliers=None, smoothOutput=None,
                  smoothResidual=None, order=2,
                  verbose=True):
    '''
    Performs a least-squares fit of a line (both slope
    and offset) for each pixel in a data cube to infer the fluxes
    and reset levels, respectively, after iterative sigma-clipping.

    d = data cube of pixel fluxes (first dimenion is read #)

    Once the fit is calculated, a model for the data is calculated,
    the residual (data - model), and the standard deviation of the
    residual.

    firstRead and or lastRead can be used to specify a subrange
    of reads to be used for the fit. Returned data cubes
    use the model evaluated for all reads.

    If smoothOutliers is set, it refers to the width of a boxcar
    smooth that should be applied iteratively to each time series
    to calculate a smooth baseline for identifying spikes, rather than
    using the residual of the line fit. Smoothoutput is an
    optional output containing the smoothed time series, and
    smoothResidual is the residual after subtracting smoothOutput

    If order is 1 the returned coeffs are (flux, reset)

    If order is > 1 the returned coeffs correspond to the order

    Return: coeffMaps, model, residual, std, mask
    '''

    x_in = np.arange(0,y_in.shape[0]).astype(float)
    
    # Make mask for outliers and specified subset of reads 
    y = y_in.copy()
    ma = np.zeros(y.shape)*np.nan
    ma[firstRead:lastRead,:,:] = 1

    if verbose: print("Fitting flux curve to reads", x_in[0] if firstRead is None else firstRead,
          "..", x_in[-1] if lastRead is None else lastRead)
    
    if smoothOutliers is not None:
        # Create the smoothing kernel of the correct width
        kernel = np.ones(smoothOutliers)       
    
    #Iterative sigma-clipping
    for i in range(maxIter+1):
        
        if smoothOutliers is None:
            # Update mask - based on line fit
            if i!=0:
                # Mask points >nSigma from model
                ind = np.where(np.logical_or(residual<-nSigma*std, residual>nSigma*std))
                ma[ind] = np.nan
            
        x = x_in[:,None,None]*ma
        y = y_in*ma
        
        if smoothOutliers is not None:
            # Update mask - based on removing smoothed baseline
            for row in range(y_in.shape[1]):
                for col in range(y_in.shape[2]):
                    rampMask = ma[:, row, col] < dummy
                    smoothed = smoothVect(y_in[:, row, col],
                                          rampMask, kernel)
                    res = y_in[:, row, col] - smoothed
                    sig = np.nanstd(res*ma[:, row, col])
                    ind = np.where(np.abs(res) > nSigma*sig)
                    ma[ind, row, col] = np.nan
                    
                    if smoothOutput is not None:
                        smoothOutput[:, row, col] = smoothed
                    
                    if smoothResidual is not None:
                        smoothResidual[:, row, col] = res
    
        if (smoothOutliers is None) or (i == maxIter):    
            # Do the fit
            
            coeffMaps = np.zeros((order, y_in.shape[1], y_in.shape[2]))
            errMaps = np.zeros(coeffMaps.shape)
            
            if order == 2:
                if verbose: print("Fitting fast linear model to ramps")
                xbar = np.nanmean(x, axis=0)
                ybar = np.nanmean(y, axis=0)

                m = np.nansum((x-xbar)*(y-ybar), axis=0) / np.nansum((x-xbar)**2, axis=0)
                b = ybar - m*xbar

                coeffMaps[0, :, :] = b
                coeffMaps[1, :, :] = m

                # here we use x_in and y_in so that we evaluate model at all reads
                model = m*x_in[:,None,None] + b
                
                # Calculate slope uncertainty
                resid = y_in - model
                n = np.sum(~np.isnan(ma), axis=0) # don't count the masked pixels
                mErr = np.sqrt(1/(n-2) * np.nansum(resid**2, axis=0) /
                               np.nansum((x - xbar)**2, axis=0))
                bErr = mErr * np.sqrt(1/n * np.nansum(x**2, axis=0))
                errMaps[0,:,:] = bErr
                errMaps[1,:,:] = mErr
            else:
                if verbose: print("Fitting polynomials to ramps of order", order)
                model = x*0
                for row in range(y_in.shape[1]):
                    for col in range(y_in.shape[2]):
                        good = np.isfinite(x[:, row, col])
                        # p = Polynomial.fit(x[good, row, col],
                        #                    y[good, row, col],
                        #                    deg=order-1)
                        if np.sum(good) > order:
                            p, cov = np.polyfit(x[good, row, col],
                                                y[good, row, col],
                                                deg=order-1, cov=True)
                        else:
                            p = np.zeros(order)*np.inf
                            cov = np.zeros((order,order))*np.inf
                        model[:, row, col] = np.polyval(p, x_in)
                        coeffMaps[:, row, col] = p[::-1]
                        errMaps[:, row, col] = np.sqrt(np.diag(cov))[::-1]
                
            residual = y_in - model
            std = np.nanstd(residual*ma, axis=0)
        
    if smoothResidual is not None:
        # Do masking in-place since user-provided array
        smoothResidual[(ma < dummy) == False] = np.nan


    return coeffMaps, errMaps, model, residual, std, ma

def calcRefMap(m, rowMed=False):
    '''
    Calculate a reference pixel map which assumes nRef pixels around the
    edges of m can be used to fit an average level for each row.
    
    Alternatively if rowMed is True, the median of the entire row
    (not just reference pixels) will be used.
    '''

    nreads = m.shape[0]
    nrows = m.shape[1]
    ncols = m.shape[2]

    #mRef = np.zeros(nrows)

    refMap = np.zeros(m.shape)
    for read in range(nreads):
        for row in range(nrows):
            
            if rowMed:
                # median of entire row
                d = m[read,row,:]
                refMap[read,row,:] = np.median(d[d < dummy])

            else:
                # median just of the ref pix
                d = np.concatenate((m[read,row,0:nRef],m[read,row,-nRef:]))
                refMap[read,row,:] = np.median(d[d < dummy])
                

                # line fit to refpix at endpoints
                #d0 = m[read,row,0:nRef]
                #d1 = m[read,row,-nRef:]
                
                #y0 = np.mean(d0[d0 < dummy])
                #y1 = np.mean(d1[d1 < dummy])

                # hack: the last refPix of one row happen 
                # close in time with first refPix of next row
                #if row > 0:
                #    y1 + (y1 + refMap[read, row-1, 0])/2.

                #slope = (y1 - y0) / ncols
                #offset = y0

                #refMap[read,row,:] = np.arange(ncols)*slope + offset
    
            
            
        # smooth mRef and use to create refMap
        #if True:
            #kernel = np.array([0.5,0.5,2.0,0.5,0.5])
            #kernel = kernel / np.sum(kernel)
            #mRefSmoothed = np.convolve(mRef, kernel, mode='same')
            #for row in range(3,nrows-3):
            #    refMap[read,row,:] = np.median(mRef[row-1:row+2]) #mRefSmoothed[row]

    return refMap


def calcLinearity(cube):
    '''
    Calculate the nonlinearity of each pixel by fitting a 2nd order polynomial
    to each ramp. Input cube should be reference pixel subtracted
    
    Input:
    cube - 3d np array data cube
    
    Return:
    the best fitting polynomial coefficients [a,b,c] (y = ax^2 + bx + c)
    '''
    
    cube -= cube[0]
    p = np.polyfit(np.arange(cube.shape[0]), cube.reshape((cube.shape[0],-1)), 2)
    p = p.reshape((3,cube.shape[1],cube.shape[2]))
    return p


def findResetAnomaly(cube, RN=50., sig=15.):
    '''
    Identify instances of the reset anomaly in a data cube.
    
    RN: read noise in ADU
    sig: deviation from linear at which reset anomaly is flagged (in multiples of RN)
    '''
    cube -= cube[0]

    # linear fit to top
    p = np.polyfit(np.arange(1,cube.shape[0]), cube[1:].reshape((cube.shape[0]-1,-1)), 1)
    p = p.reshape((2,cube.shape[1],cube.shape[2]))

    x = np.arange(cube.shape[0])
    mod = p[0]*(x[:,None,None] * np.ones(cube.shape)) + p[1]
    resid = cube - mod
    
    return (resid[0]/RN) < -sig
            

def smoothMap(m, good, kernel):
    '''
    Calculate smooth version of a map using supplied kernel, but only
    considering good pixels

    Inputs:
    m - map (should not have any nan)
    good - which pixels are good
    kernel - smoothing kernel

    Return:
    smoothed map
    '''

    nRows = m.shape[0]
    nCols = m.shape[1]

    # create a "variance" map that massively downweights previously
    # flagged pixels
    varMap = np.zeros((nRows, nCols)) + dummy
    varMap[good] = 1.

    # calculate smoothed good reset level map using kernel and varmap
    numerator = signal.fftconvolve(m / varMap, kernel, mode='same')
    denominator = signal.fftconvolve(1. / varMap, kernel**2, mode='same')

    smoothed = numerator / denominator

    return smoothed

def smoothVect(v, good, kernel):
    '''
    Calculate smooth version of a vector using supplied kernel, but only
    considering good pixels

    Inputs:
    v - vector (should not have any nan)
    good - which pixels are good
    kernel - smoothing kernel

    Return:
    smoothed vector
    '''

    n = v.shape[0]
    
    # create a "variance" vector that massively downweights previously
    # flagged pixels
    varVect = np.zeros(n) + dummy
    varVect[good] = 1.

    # calculate smoothed good reset level map using kernel and varVect
    numerator = signal.correlate(v / varVect, kernel, mode='same')
    denominator = signal.correlate(1. / varVect, kernel**2, mode='same')

    smoothed = numerator / denominator

    return smoothed

# --- ODGW class -------------------------------------------------------------

class Odgw:
    '''
    Class for working with On-Detector Guide Window (ODGW) data
    '''

    def __init__(self, fileName, skipReads=0, lastRead=None, version=None, verbose=True,
                 forceColSwapAndInvert: Optional[bool] = None):
        '''
        Load FITS file containing raw data into Odgw object.
        
        As soon as the data are loaded we calculate the complement,
        
        data = 65535 - unsignedData
        
        so that the reset values are closer to zero, and as we
        accumulate flux the values are more positive.
        
        If skipReads is specified, strip off the requested number
        of reads from the start (e.g., an initial sacrificial
        read).
        
        If lastRead is specified, strip off reads after lastRead
        (and it is counted from the start, i.e., prior to application
        of skipReads).
        
        TODO: version
        
        The returned dictionary does not have the full-frame
        and window data separated. For that operation call
        separateWindow using the returned dict from this function
        as the argument.
        '''

        # --- initialize member variables ---

        # Minimally processed data from the FITS file
        self.data = None
        self.hdr = None

        # FITS header data
        self.NAXIS1 = None
        self.NAXIS2 = None
        self.DATE = None
        self.NFFREADS = None
        self.XFRAMESZ = None
        self.YFRAMESZ = None
        self.XWINSTRT = None # old
        self.YWINSTRT = None # old
        self.XWINSTOP = None # old
        self.YWINSTOP = None # old
        self.WINNPIX = None
        self.FRSTINT = None
        self.WRSTINT = None
        self.WREADINT = None

        self.haveWin = None
        self.haveFull = None
        self.skipReads = None

        self.YFRM = None

        self.NWINS = None
        self.XWIN = None
        self.YWIN = None
        self.XWINSIZE = None
        self.YWINSIZE = None

        self.winReadsPerFrame = None
        self.firstCleanWin = None

        # separated full-frame and window data
        self.ff = None
        self.ffDiffs = None
        self.win = None

        # flux modeling
        self.coeffMaps = None
        self.errMaps = None
        self.model = None
        self.residual = None
        self.std = None
        self.flux = None
        self.fluxErr = None

        self.winCoeffMaps = None
        self.winErrMaps = None
        self.winModel = None
        self.winResidual = None
        self.winStd = None
        self.winFlux = None
        self.winFluxErr = None
        
        self.winSmooth = None
        self.winSmoothResidual = None
        
        self.winRampStarts = None

        # good pixel mask (nan where bad, 1 otherwise)
        self.pixMask = None
        self.winPixMask = None

        # dark (scaled for the number of reads)
        self.dark = None
        
        # map/cube of background noise using reference pixes
        self.refMap = None

        # parsed keywords to be added to full-frame output file headers
        self.ffHdr = None
        
        # electron conversion gain in e-/ADU
        self.egain = 1.
        # RMS read noise in e-
        self.rnoise = 0.

        # --- load file into the object and perform checks --- 

        self.skipReads = skipReads

        if verbose: print("Loading", fileName)

        # load in the FITS file
        f = fits.open(fileName)
        self.hdr = f[0].header
        
        tempData = f[0].data.astype("int64")
        #self.data = f[0].data.astype("int64")
        # data_uint = 65535 - f[0].data
        # tempData = data_uint.astype("int64")

        # Decide if we need to swap alternating columns
        swapColsAndInvert = False
        if forceColSwapAndInvert is not None:
            # Caller is forcing us to do it a certain way
            swapColsAndInvert = forceColSwapAndInvert
        elif 'DATE' in self.hdr:
            # Try to figure it out from the exposure date
            refDate = datetime(2023,12,20,14,28)  # see commit 3dc616bde
            exposureDate = datetime.fromisoformat(self.hdr['DATE'])
            if exposureDate < refDate:
                swapColsAndInvert = True
        else:
            # Probably very old data if no 'DATE' in hdr...
            swapColsAndInvert = True
        
        if swapColsAndInvert:
            print("Swapping columns and inverting")
            self.data = tempData*0
            for r in range(self.data.shape[0]):
                for colPair in range(self.data.shape[2]//2):
                    self.data[r, :, colPair*2] = 65535 - tempData[r, :, colPair*2 + 1]
                    self.data[r, :, colPair*2 + 1] = 65535 - tempData[r, :, colPair*2]
        else:
            self.data = tempData
            
        if (skipReads > 0) or (lastRead is not None):
            if lastRead is None:
                last = self.data.shape[0]
            else:
                last = lastRead
            
            self.data = self.data[skipReads:lastRead,:,:]
        self.NFFREADS = self.data.shape[0] #self.hdr['NAXIS3'] - skipReads

        # get metadata from header
        fullHdr = False   # check for complete header info
        haveWin = False   # guide window data are present
        haveFull = False    # full-frame data are present
        try:
            self.NAXIS1 = self.hdr['NAXIS1']
            self.NAXIS2 = self.hdr['NAXIS2']
            self.DATE = self.hdr['DATE']
            self.XFRAMESZ = self.hdr['XFRAMESZ']
            self.YFRAMESZ = self.hdr['YFRAMESZ']
            self.XWINSTRT = self.hdr['XWINSTRT']
            self.YWINSTRT = self.hdr['YWINSTRT']
            self.XWINSTOP = self.hdr['XWINSTOP']
            self.YWINSTOP = self.hdr['YWINSTOP']
            try:
                self.FRSTINT = self.hdr['FRSTINT']
            except Exception as e:
                print("No FRSTINT keyword, setting to 0")
                self.FRSTINT = 0
            # try:
            #     self.WRSTINT = self.hdr['WRSTINT']
            # except Exception as e:
            #     print("No WRSTINT keyword, setting to 0")
            #     self.WRSTINT = 0
            
            # Kludgey way to figure out if any window data
            if (self.XFRAMESZ == self.NAXIS1):
                haveWin = False
            else:
                haveWin = True

            # get starting row of full frame, backward compatible
            if 'YFRMSTRT' in self.hdr:
                self.YFRM = self.hdr['YFRMSTRT']
            else:
                self.YFRM = 0

            # extract (potentially multiple) window parameters
            if 'NWINS' in self.hdr:
                self.NWINS = self.hdr['NWINS']
                self.XWINSIZE = self.hdr['XWINSIZE']
                self.YWINSIZE = self.hdr['YWINSIZE']

                # the *WINSTRT are duplicated by *WIN0 in newer
                # data. This way is backward compatible
                self.XWIN = [self.hdr['XWINSTRT']]
                self.YWIN = [self.hdr['YWINSTRT']]
                try:
                    self.WRSTINT = [self.hdr['WRSINT0']]
                except Exception as e:
                    print("No WRSINT keyword(s), setting to 0")
                    self.WRSTINT = [0]
                for win in range(1, self.NWINS):
                    self.XWIN.append(self.hdr['XWIN'+str(win)])
                    self.YWIN.append(self.hdr['YWIN'+str(win)])
                    try:
                        self.WRSTINT.append(self.hdr['WRSINT'+str(win)])
                    except Exception as e:
                        print("No WRSINT keyword(s), setting to 0")
                        self.WRSTINT.append(0)
                self.XWIN = np.array(self.XWIN)
                self.YWIN = np.array(self.YWIN)

            elif haveWin:
                # hack for older data when only 1 window was possible
                self.NWINS = 1
                self.XWINSIZE = (self.XWINSTOP - self.XWINSTRT) + 1
                self.YWINSIZE = (self.YWINSTOP - self.YWINSTRT) + 1
                self.XWIN = np.array([self.XWINSTRT])
                self.YWIN = np.array([self.YWINSTRT])
            else:
                self.NWINS = 0

            fullHdr = True
            
            if 'WREADINT' in self.hdr:
                # Not reading window every row
                self.WREADINT = self.hdr['WREADINT']

                if self.WREADINT == 0:
                    self.WREADINT = 1
                    if self.NWINS > 0:
                        self.winReadsPerFrame = self.YFRAMESZ // self.NWINS
                elif self.YFRAMESZ % self.WREADINT != 0:
                    raise Exception("YFRAMESZ is not divisible by WREADINT")
                else:
                    self.winReadsPerFrame = (self.YFRAMESZ // self.NWINS // self.WREADINT)
            else:
                self.WREADINT = 1
                if self.NWINS > 0:
                    self.winReadsPerFrame = self.YFRAMESZ // self.NWINS
            
            if haveWin:
                self.WINNPIX = self.XWINSIZE*self.YWINSIZE
            
            if (self.YFRAMESZ != self.NAXIS2):
                raise Exception("Problem with file? YFRAMESZ != NAXIS2")
            
            if (self.XFRAMESZ == 0) or (self.YFRAMESZ == 0):
                haveFull = False
            else:
                haveFull = True
            
            # write parsed header keywords for ffHdr for later use
            ffHdr = {}
            ffHdr['DATE'] = self.DATE
            ffHdr['FRSTINT'] = self.FRSTINT
            ffHdr['XFRAMESZ'] = self.XFRAMESZ
            ffHdr['YFRAMESZ'] = self.YFRAMESZ
            ffHdr['NWINS'] = self.NWINS
            if self.NWINS > 0:
                ffHdr['WREADINT'] = self.WREADINT
                # ffHdr['WRSTINT'] = self.WRSTINT
                ffHdr['XWINSIZE'] = self.XWINSIZE
                ffHdr['YWINSIZE'] = self.YWINSIZE
                for win in range(self.NWINS):
                    ffHdr['XWIN'+str(win)] = self.XWIN[win]
                    ffHdr['YWIN'+str(win)] = self.YWIN[win]
                    ffHdr['WRSINT'+str(win)] = self.WRSTINT[win]
            self.ffHdr = ffHdr

        except Exception as e:
            print("Failed to read full header from %s, continuing anyway: " % fileName, e)
       
        self.haveWin = haveWin
        self.haveFull = haveFull

        if fullHdr and verbose:
            print('Have full header:')
            print('Skip initial reads=', self.skipReads)
            print('NFFREADS=', self.NFFREADS)
            print('XFRAMESZ=', self.XFRAMESZ)
            print('YFRAMESZ=', self.YFRAMESZ)
            print('FRSTINT=', self.FRSTINT)
            # print('WRSTINT=', self.WRSTINT)
            print('WREADINT=', self.WREADINT)

            if self.NWINS is not None:
                print('NWINS=', self.NWINS)
                print('XWINSIZE=', self.XWINSIZE)
                print('YWINSIZE=', self.YWINSIZE)
                print('WINNPIX=', self.WINNPIX)
                for i in range(self.NWINS):
                    print('XWIN'+str(i)+'=', self.XWIN[i])
                    print('YWIN'+str(i)+'=', self.YWIN[i])
                    print('WRSINT'+str(i)+'=', self.WRSTINT[i])
                    
        f.close()
        
    def linearize(self, poly):
        '''
        Linearize the image using the supplied polynomial coefficients.
        Correction is independent of x-axis used in polynomial fit.
        '''
        
        reset = self.data[0].copy()
        
        if self.haveWin:
            # Extend the poly map to match the window reads on the right side
            pwins = []
            for winnum in range(self.NWINS):
                # create one row of coeffs for each window
                r = self.YWIN[winnum]-self.YFRM
                c = self.XWIN[winnum]
                pwins.append(poly[:,r:r+self.YWINSIZE,c:c+self.XWINSIZE].reshape((3,-1)))
                
            pwins = np.transpose(pwins, (1,0,2)) # coeff axis first
            # repeat to fill YFRAMESZ -- do an extra than truncate to frame size
            pwins = np.tile(pwins, (1,self.YFRAMESZ//self.NWINS+1,1))[:,:self.YFRAMESZ]
                
            poly = np.append(poly,pwins,axis=2)
            
            # Find the correct window resets in the raw data
            winresets = []
            for winnum in range(self.NWINS):
                # Get when row is reset
                resetat = (np.arange(self.WINNPIX)//self.XWINSIZE + self.YWIN[winnum]-self.YFRM)
                # Correct to first time win is read after reset
                resetat += (resetat-winnum)%self.NWINS
                # Get the reset value
                res = np.array([self.data[0,resetat[i],self.XFRAMESZ+i] for i in range(self.WINNPIX)])
                winresets.append(res)
                
            winresets = np.tile(winresets, (self.YFRAMESZ//self.NWINS+1,1))[:self.YFRAMESZ]
            reset[:,self.XFRAMESZ:] = winresets
        
        self.data -= reset
        poly[np.abs(poly)<1e-2] = 0
        correction = -1/(4*poly[0]) * (-poly[1] + np.sqrt(poly[1]**2 - 4*poly[0]*poly[2] + 4*poly[0]*self.data))**2
        correction[np.isnan(correction)] = 0
        correction[np.abs(correction)>20000] = 0
        self.data = self.data + correction
        self.data += reset
        

    def separateWindow( self, subFfReset=True ):
        '''
        Separate full-frame data from the guide window data,
        and store in the object as members ff and win
        'ff' and 'win'.
        
        If subFfReset is true, subtract off the most recent full-frame reset
        '''
        
        # separate the full-frame and window reads to separate data cubes, and
        # also calculate the diffs
        if self.haveFull:
            fullReads = np.zeros( (self.NFFREADS, self.YFRAMESZ, self.XFRAMESZ) )
        
        if self.haveWin:
            nWinReads = self.NFFREADS * self.winReadsPerFrame
            winReads = np.zeros( (self.NWINS, nWinReads, self.YWINSIZE, self.XWINSIZE) )

        # Initialize most recent full frame reset index
        ffRef = 0

        if subFfReset:
            ffDiffs = np.zeros( (self.NFFREADS-1, self.YFRAMESZ, self.XFRAMESZ) )
            ffDiffIndex = 0

        for read in range(self.NFFREADS):
            
            if self.FRSTINT > 0:
                # Update reset index if needed
                if read % (self.FRSTINT+1) == 0:
                    ffRef = read
            
            if self.haveFull:
                fullReads[read,:,:] = self.data[read,0:self.YFRAMESZ,0:self.XFRAMESZ]
                
                if subFfReset and (read != ffRef):
                    ffDiffs[ffDiffIndex] = fullReads[read,:,:] - fullReads[ffRef,:,:]
                    ffDiffIndex = ffDiffIndex + 1
                                                
            if self.haveWin:
                # outer loop over window
                for win in range(self.NWINS):
                    # inner loop over particular window reads within current
                    # full-frame read
                    firstWinRead = read*self.winReadsPerFrame
                    for winRead in range(firstWinRead, firstWinRead+self.winReadsPerFrame):
                        # identify the FF row that has this read of the particular window
                        ffRow = win + winRead*self.NWINS*self.WREADINT - read*self.YFRAMESZ
                        winReads[win,winRead,:,:] = np.reshape(
                                                    self.data[read,
                                                              ffRow,
                                                              self.XFRAMESZ:self.XFRAMESZ+self.WINNPIX],
                                                    (self.YWINSIZE, self.XWINSIZE) )

            if self.haveFull:
                self.ff = fullReads
                
                if subFfReset:
                    self.ffDiffs = ffDiffs

            if self.haveWin:
                self.win = winReads

    def subDark(self, dark, isFlux=False):
        '''
        The supplied dark should be a frame-by-frame np array match of
        the odgw exposure, including windows.
        
        Alternatively, if 'isFlux' is True, the dark is assumed to be
        a flux measurement from a dark exposure. It is multiplied by
        the number of reads in this exposure and then subtracted from
        the data.
        '''
        
        self.dark = dark
        
        if isFlux:
            if self.ff is None:
                raise Exception("separateWindow() must be called before subDark() if isFlux=True")
            
            newMask = np.copy(dark)
            newMask[newMask > 0] = 1.0
            if self.pixMask is None:
                self.pixMask = newMask
            else:
                self.pixMask = self.pixMask * newMask
            
            for read in range(self.NFFREADS):
                self.ff[read,:,:] = self.ff[read,:,:] - read*self.dark
                
            #TODO: subtract window flux
            
        else:
            if self.ff is not None:
                raise Exception("separateWindow() cannot called before subDark() if isFlux=False")
            
            # Verify dark has correct shape
            if dark.shape != self.data.shape:
                raise Exception("Supplied dark shape does not match data")
            
            self.data = self.data.astype(np.float64) - dark
            


    def fitModel(self, firstRead=None, lastRead=None, method='OLS', order=2, verbose=True):
        '''
        Fit (linear) flux model to full-frame pixel measurements
        '''

        self.coeffMaps, self.errMaps, self.model, self.residual, self.std, self.readMask = calcFluxModel(self.ff, firstRead=firstRead, lastRead=lastRead, order=order, verbose=verbose)
        
        # Use cadence from the header to get flux in e-/second
        if 'FTIME' in self.hdr:
            framePer = self.hdr['FTIME'] / 1e6
        else:
            framePer = self.YFRAMESZ * self.hdr['ROWPER'] / 1e6 # seconds/frame
        self.flux = self.egain * self.coeffMaps[1] / framePer # e-/second
        
        if method=='Robberto':
            fullMaps,_,_,_,_,_ = calcFluxModel(self.ff+self.dark[:,:,:self.XFRAMESZ],
                                               firstRead=firstRead, lastRead=lastRead, order=order, verbose=verbose)
            fullflux = self.egain * fullMaps[1] / framePer
            n = self.ff.shape[0]
            var_P = 6/5 * (n**2 + 1)/(n**3 - n) * fullflux          # Poisson noise
            var_R = 12*self.rnoise**2 / (n**3 - n) / framePer**2    # read noise
            self.fluxErr = np.sqrt(var_P+var_R)
        else:
            # use the OLS error
            self.fluxErr = self.egain * self.errMaps[1] / framePer
        

    def fitWinModel(self, firstRead=None, lastRead=None, smoothOutliers=None, method='OLS', order=2, verbose=True):
        '''
        Fit (linear) flux model to window reads, accounting for full-frame reset that is assumed to occur as
        part of the first full-frame read.
        '''

        # identify the full-frame row at which the initial reset will have
        # swept past each window
        firstCleanWinRow = self.YWIN + self.YWINSIZE - self.YFRM
        
        # And then the first read that will be clean for each window
        self.firstCleanWin = (np.ceil(firstCleanWinRow / self.WREADINT / self.NWINS)).astype(int)
        
        self.winCoeffMaps = []
        self.winErrMaps = []
        self.winModel = []
        self.winResidual = []
        self.winStd = []
        self.winReadMask = []
        self.winFlux = []
        self.winFluxErr = []
        
        self.winRampStarts = []
        
        if smoothOutliers is not None:
            self.winSmooth = []
            self.winSmoothResidual = []

        if smoothOutliers is not None:
            smoothOutput = np.zeros(self.win.shape)
            smoothResidual = np.zeros(self.win.shape)
        else:
            smoothOutput = None
            smoothResidual = None
            
        nWinReads = self.win.shape[1]
        
        if method=='Robberto':
            # prepare the dark data - separate the window
            # This assumes the window config of the dark is exactly that of self
            darkWin = np.zeros( (self.NWINS, nWinReads, self.YWINSIZE, self.XWINSIZE) )
            for read in range(self.NFFREADS):
                # outer loop over window
                for win in range(self.NWINS):
                    # inner loop over particular window reads within current
                    # full-frame read
                    firstWinRead = read*self.winReadsPerFrame
                    for winRead in range(firstWinRead, firstWinRead+self.winReadsPerFrame):
                        # identify the FF row that has this read of the particular window
                        ffRow = win + winRead*self.NWINS*self.WREADINT - read*self.YFRAMESZ
                        darkWin[win,winRead,:,:] = np.reshape(
                                                    self.dark[read,
                                                            ffRow,
                                                            self.XFRAMESZ:self.XFRAMESZ+self.WINNPIX],
                                                    (self.YWINSIZE, self.XWINSIZE) )
                        
        # Use cadence from the header to get flux in e-/second
        if 'RTIME' in self.hdr:
            rowPer = self.hdr['RTIME'] / 1e6
        else:
            rowPer = self.hdr['ROWPER'] / 1e6
        winPer = self.NWINS * self.WREADINT * rowPer # seconds/window read

        for winNum in range(self.NWINS):
            # Fit a new ramp every time the window is reset
            wrstint = self.WRSTINT[winNum]
            if wrstint == 0:
                # If window is never reset, start at firstCleanWin
                rampStarts = [self.firstCleanWin[winNum]]
            else:
                rampStarts = np.arange(0, nWinReads, wrstint)
                # Fit only ramps starting after firstCleanWin
                rampStarts = rampStarts[rampStarts >= self.firstCleanWin[winNum]]
                
            self.winRampStarts.append(rampStarts)
                
            winCoeffMaps = []
            winCoeffErrs = []
            winModel = np.zeros((0, self.YWINSIZE, self.XWINSIZE))
            winResidual = np.zeros((0, self.YWINSIZE, self.XWINSIZE))
            winStd = np.zeros((0, self.YWINSIZE, self.XWINSIZE))
            winReadMask = np.zeros((0, self.YWINSIZE, self.XWINSIZE))
                
            for i,rs in enumerate(rampStarts):
                rampEnd = np.min((nWinReads,rs+wrstint))
                if rampEnd==self.firstCleanWin[winNum]: rampEnd = nWinReads
                
                if smoothOutliers:
                    rampCoeffMaps, coeffErr, rampModel, rampResidual, rampStd, rampReadMask = \
                        calcFluxModel(self.win[winNum, rs:rampEnd, :, :], firstRead=firstRead, lastRead=lastRead,
                        smoothOutliers=smoothOutliers,
                        smoothOutput=smoothOutput[winNum, rs:rampEnd, :, :],
                        smoothResidual=smoothResidual[winNum, rs:rampEnd, :, :],
                        order=order, verbose=verbose)
                else:
                    rampCoeffMaps, coeffErr, rampModel, rampResidual, rampStd, rampReadMask = \
                        calcFluxModel(self.win[winNum, rs:rampEnd, :, :], firstRead=firstRead, lastRead=lastRead,
                                      order=order, verbose=verbose)
                        
                if method=='Robberto':
                    fullMaps,_,_,_,_,_ = calcFluxModel((self.win+darkWin)[winNum, rs:rampEnd, :, :],
                                                       firstRead=firstRead, lastRead=lastRead, order=order, verbose=verbose)
                    fullflux = self.egain * fullMaps[1] / winPer
                    n = rampEnd - rs
                    var_P = 6/5 * (n**2 + 1)/(n**3 - n) * fullflux      # Poisson noise
                    var_R = 12*self.rnoise**2 / (n**3 - n) / winPer**2  # read noise
                    coeffErr[1] = winPer*np.sqrt(var_P+var_R)/self.egain
                        
                # Fit coefficients have an additional dimension for the coeffs of each ramp,
                # but winModel, winResidual, etc are just concatenated to make one function
                # across all ramps
                winCoeffMaps.append(rampCoeffMaps)
                winCoeffErrs.append(coeffErr)
                winModel = np.append(winModel, rampModel, axis=0)
                winResidual = np.append(winResidual, rampResidual, axis=0)
                winStd = np.append(winStd, rampStd[None,:,:], axis=0)
                winReadMask = np.append(winReadMask, rampReadMask, axis=0)
                
                
            # Apply the per-read outlier mask calculated above
            self.win[winNum, rampStarts[0]:,:,:] *= winReadMask
            winResidual *= winReadMask
            
            self.winCoeffMaps.append(winCoeffMaps)
            self.winErrMaps.append(winCoeffErrs)
            self.winModel.append(winModel)
            self.winResidual.append(winResidual)
            self.winStd.append(winStd)
            self.winReadMask.append(winReadMask)
            
            self.winFlux.append([self.egain * map[1] / winPer for map in winCoeffMaps])
            self.winFluxErr.append([self.egain * err[1] / winPer for err in winCoeffErrs])
            
            if smoothOutliers is not None:
                self.winSmooth.append(smoothOutput[winNum,rampStarts[0]:,:,:])
                self.winSmoothResidual.append(smoothResidual[winNum,rampStarts[0]:,:,:])
        

    def recombine(self):
        '''
        Recombine separated window flux into the full frame.
        '''
        
        for winNum in range(self.NWINS):
            # If window has been reset, average the fluxes weighted by uncertainty
            # avWinFlux = np.nanmean(self.winFlux[winNum], axis=0)
            # avWinErr = np.nanmean(self.winFluxErr[winNum], axis=0)
            f = np.array(self.winFlux[winNum])
            ferr = np.array(self.winFluxErr[winNum])
            avWinFlux = np.nansum(f/ferr**2, axis=0) / np.nansum(1/ferr**2, axis=0)
            avWinErr = 1 / np.sqrt(np.nansum(1/ferr**2, axis=0))
            
            winSlice = np.s_[self.YWIN[winNum]-self.YFRM:self.YWIN[winNum]+self.YWINSIZE-self.YFRM,
                             self.XWIN[winNum]:self.XWIN[winNum]+self.XWINSIZE]
            
            self.flux[winSlice] = avWinFlux
            self.fluxErr[winSlice] = avWinErr
            
            # Keep mask information
            if self.pixMask is not None:
                self.flux *= self.pixMask
                self.fluxErr *= self.pixMask
    
    
    def writeCubes(self, prefix):
        '''
        Write cubes from the odgw object to FITS files.
        
        Gives the files the supplied prefix, and depending on
        what is present in the object:
        
        [prefix]_raw.fits
        [prefix]_fullReads.fits
        [prefix]_fullDiffs.fits
        [prefix]_winReads.fits
        
        Note that "raw" is just the original data written by the
        ODGW code, but setting it to (65535 - data) so that values
        increase as flux is accumulated.
        '''
        
        #hdu = fits.PrimaryHDU(self.data)
        #hdul = fits.HDUList([hdu])
        #hdul.writeto(prefix+'_raw.fits', overwrite=True, extraHdr=self.ffHdr) 
        writeFits( self.data, prefix+'_raw.fits', overwrite=True, extraHdr=self.ffHdr)

        if self.ff is not None:
            writeFits( self.ff, prefix+'_fullReads.fits', overwrite=True, extraHdr=self.ffHdr)
            
        if self.ffDiffs is not None:
            writeFits( self.ffDiffs, prefix+'_fullDiffs.fits', overwrite=True, extraHdr=self.ffHdr)

        if self.coeffMaps is not None:
            writeFits(self.coeffMaps, prefix+'_coeffs.fits', overwrite=True, extraHdr=self.ffHdr)    

        if self.model is not None:
            writeFits( self.model, prefix+'_dataModel.fits', overwrite=True, extraHdr=self.ffHdr)

        if self.residual is not None:
            writeFits( self.residual, prefix+'_residual.fits', overwrite=True, extraHdr=self.ffHdr)

        if self.std is not None:
            writeFits( self.std, prefix+'_stdev.fits', overwrite=True, extraHdr=self.ffHdr)

        if  self.win is not None:
            for win in range(self.NWINS):
                winHdr = {
                    'DATE' : self.DATE,
                    'WINID': win,
                    'XWIN': self.XWIN[win],
                    'YWIN': self.YWIN[win],
                    'XWINSIZE': self.XWINSIZE,
                    'YWINSIZE': self.YWINSIZE,
                    'WRSTINT': self.WRSTINT[win],
                    'WREADINT': self.WREADINT,
                    'CLEANRD': self.firstCleanWin[win]
                }   
                writeFits( self.win[win,:,:,:], prefix+'_winReads'+str(win)+'.fits',
                            overwrite=True, extraHdr=winHdr)

        if self.winCoeffMaps is not None:
            for win in range(self.NWINS):
                winHdr = {
                    'DATE' : self.DATE,
                    'WINID': win,
                    'XWIN': self.XWIN[win],
                    'YWIN': self.YWIN[win],
                    'XWINSIZE': self.XWINSIZE,
                    'YWINSIZE': self.YWINSIZE,
                    'WRSTINT': self.WRSTINT[win],
                    'WREADINT': self.WREADINT
                }      
                writeFits( self.winCoeffMaps[win], prefix+'_winCoeffs'+str(win)+'.fits',
                            overwrite=True, extraHdr=winHdr)
                writeFits( self.winModel[win], prefix+'_winModel'+str(win)+'.fits',
                            overwrite=True, extraHdr=winHdr)
                writeFits( self.winResidual[win], prefix+'_winResidual'+str(win)+'.fits',
                            overwrite=True, extraHdr=winHdr)
                writeFits( self.winStd[win], prefix+'_winStdev'+str(win)+'.fits',
                            overwrite=True, extraHdr=winHdr)
                
                if self.winSmooth is not None:
                    writeFits( self.winSmooth[win], prefix+'_winSmooth'+str(win)+'.fits',
                                overwrite=True, extraHdr=winHdr)
                
                if self.winSmoothResidual is not None:
                    writeFits( self.winSmoothResidual[win], prefix+'_winSmoothResidual'+str(win)+'.fits',
                                overwrite=True, extraHdr=winHdr)


    def calcMask(self, resetPreClip=45000, sigReset=4, sigFlux=4,
                 localClipResetRad=10, localClipFluxRad=10,
                 fluxResidLimit=5000, prefix=None, verbose=True):
        '''
        A more sophisticated method for calculating the good pixel mask
        
        Rejects outliers using both the reset levels and fluxes inferred
        from the linear fits to the reads
          - an iterative sigma clip on the reset level (change with sigReset)
          - an iterative sigma clip on the flux (change with sigFlux)

        Handles the reference pixels separately from the main pixels
        
        The main (light-sensitive) pixels use smooth maps (kernel radii given by
        localClipResetRad / localClipFluxRad) when calculating residuals for outliers

        An initial clip is performed on reset levels greater than resetPreClip
        
        '''
        
        # Going to do this 4 times, once for each of the 4 border rectangles
        # of reference pixels, and finally the photosensitive pixels

        boxes = []

        # bottom ref pixels including corners - only if ref rows are in the frame
        refBot = 0
        if self.YFRM == 0: 
            boxes.append({'x':[0, self.XFRAMESZ], 'y':[0, nRef]})
            refBot = nRef
        
        # top ref pixels including corners - only if ref rows are in the frame
        refTop = 0
        if self.YFRM + self.YFRAMESZ == 2048:
            boxes.append({'x':[0, self.XFRAMESZ], 'y':[self.YFRAMESZ-nRef, self.YFRAMESZ]})
            refTop = nRef
        
        # left ref pixels excluding corners
        boxes.append({'x':[0, nRef], 'y':[refBot, self.YFRAMESZ-refTop]})
        
        # right ref pixels excluding corners
        boxes.append({'x':[self.XFRAMESZ-nRef, self.XFRAMESZ], 'y':[nRef, self.YFRAMESZ-refTop]})
        
        # light sensitive pixels
        boxes.append({'x':[nRef, -nRef], 'y':[refBot, self.YFRAMESZ-refTop]})
        lsPixels = boxes[-1]

        mask = np.zeros(self.coeffMaps.shape[1:3])

        for box in boxes:
            if verbose: print("Calculating mask for box", box)
            xlim = box['x']
            ylim = box['y']

            resetData = np.copy(self.coeffMaps[0, ylim[0]:ylim[1], xlim[0]:xlim[1]])
            fluxData = np.copy(self.coeffMaps[1, ylim[0]:ylim[1], xlim[0]:xlim[1]])

            nRows = resetData.shape[0]
            nCols = resetData.shape[1]

            if verbose: print("pre-clip large reset outliers")
            good = resetData < resetPreClip
            
            if box == lsPixels:
                #
                # Fancy masking for light-sensive pixels
                #
                
                # create a circular kernel that will calculate the average value
                # of pixels within a certain radius
                y,x = np.mgrid[:nRows,:nCols]
                r = np.sqrt((y-nRows/2)**2 + (x-nCols/2)**2) 
                
                # disc shaped kernel
                kernelReset = r <= localClipResetRad
                kernelFlux = r <= localClipFluxRad

                # Gaussian kernel
                #kernelReset = 2.*np.exp(-0.5*(r/localClipResetRad)**2)                
                #kernelFlux = 2.*np.exp(-0.5*(r/localClipFluxRad)**2)
                
                # Normalize the kernel?
                #kernelReset = kernelReset/np.sum(kernelReset)
                #kernelFlux = kernelFlux/np.sum(kernelFlux)

                # iteratively flag reset outliers
                for iter in range(10):
                    smoothReset = smoothMap(resetData, good, kernelReset)

                    # calculate the standard deviation of pixels after subtracting off
                    # a smoothed map that only considers good pixels
                    resetResidual = resetData - smoothReset
                    stdevResetResidual = np.std( resetResidual[good] )

                    if verbose:
                        print("Reset residual stdev:",stdevResetResidual)
                   
                    # local stdev of residual within kernel aperture
                    localStdevReset = smoothMap(resetResidual**2, good, kernelReset)*np.sum(kernelReset)
                    localStdevReset = np.sqrt(localStdevReset / np.sum(kernelReset))

                    # update list of good pixels
                    #good = np.logical_and(good, np.abs(resetResidual) < sigReset*stdevResetResidual)
                    good = np.logical_and(good, np.abs(resetResidual) < sigReset*localStdevReset)
                    
                # iterative flag flux outliers
                for iter in range(10):
                    smoothFlux = smoothMap(fluxData, good, kernelFlux)

                    # calculate the standard deviation of pixels after subtracting off
                    # a smoothed map that only considers good pixels
                    fluxResidual = fluxData - smoothFlux
                    stdevFluxResidual = np.std( fluxResidual[good] )

                    if verbose:
                        print("Flux residual stdev:",stdevFluxResidual)

                     # local stdev of residual within kernel aperture
                    localStdevFlux = smoothMap(fluxResidual**2, good, kernelFlux)*np.sum(kernelFlux)
                    localStdevFlux = np.sqrt(localStdevFlux / np.sum(kernelFlux))

                    # update list of good pixels
                    #good = np.logical_and(good, np.abs(fluxResidual) < sigFlux*stdevFluxResidual)
                    #good = np.logical_and(good, np.abs(fluxResidual) < sigFlux*localStdevFlux)
                
                    # only clip positive outliers for flux
                    good = np.logical_and(good, fluxResidual < sigFlux*localStdevFlux)

                    # also clip absolute positive outliers in residual
                    good = np.logical_and(good, fluxResidual < fluxResidLimit)

                
                bad = np.logical_not(good)
                smoothReset[bad] = np.nan
                resetResidual[bad] = np.nan
                smoothFlux[bad] = np.nan
                fluxResidual[bad] = np.nan
                
                # writeFits(smoothReset, 'test_reset.fits', overwrite=True)
                # writeFits(resetResidual, 'test_resetResidual.fits', overwrite=True)
                # writeFits(smoothFlux, 'test_flux.fits', overwrite=True)
                # writeFits(fluxResidual, 'test_fluxResidual.fits', overwrite=True)
                # writeFits(fluxResidual/localStdevFlux, 'test_fluxSNR.fits', overwrite=True)


                # set mask pixels in this box
                mask[ylim[0]:ylim[1], xlim[0]:xlim[1]] = 1.
                mask[ylim[0]:ylim[1], xlim[0]:xlim[1]][bad] = np.nan

            else:
                #
                # Simple masking for reference pixels
                #
                
                resetData[resetData > resetPreClip] = np.nan
                remaining = resetData[~np.isnan(resetData)]
            
                if verbose:
                    print("perform iterative sigma clip on reset to remove remaining outliers..")
                c, lower, upper = stats.sigmaclip(remaining, low=sigReset, high=sigReset)
                if verbose: print(lower,upper)
                resetData[resetData < lower] = np.nan
                resetData[resetData > upper] = np.nan
                resetData[resetData < dummy] = 1.
                    
                remaining = fluxData[~np.isnan(fluxData)]
                
                if verbose:
                    print("perform iterative positive sigma clip on flux to remove remaining outliers..")
                c, lower, upper = stats.sigmaclip(remaining, low=3.0, high=3.0)
                if verbose: print(lower,upper)
                #fluxData[fluxData < lower] = np.nan
                fluxData[fluxData > upper] = np.nan
                fluxData[fluxData < dummy] = 1.
                
                # Set mask pixels corresponding to this box
                mask[ylim[0]:ylim[1], xlim[0]:xlim[1]] = resetData * fluxData
            
        # Finished calculating mask so apply to all of the data
        self.setPixMask(mask, prefix)
        
    def calcMask2(self, clip=3, dummyN=1000, maxIter=20, prefix=None, verbose=True):
        """Masking method based on the expected distributions of values"""
        res = self.coeffMaps[0]
        flux = self.coeffMaps[1]
        
        good = np.ones(res.shape).astype(bool)
        mask = np.ones(res.shape)

        # Start with reset values, clip until the mean/stdv of the reset matches that of
        # the best-fitting Gaussian of the histogram of reset values
        for i in range(maxIter):
            mu = np.nanmean(res*mask)
            sig = np.nanstd(res*mask)

            vals, bins = np.histogram(res[good], bins=50, density=True)
            bins = (bins[1:] + bins[:-1]) / 2
            (fitmu,fitsig),_ = curve_fit(norm.pdf, bins, vals, p0=(21000,1000))

            diff = np.abs(mu-fitmu)
            # dummyN biases the result a bit, but I found 1000 to be a good value
            stdofmean = np.sqrt(sig**2/dummyN + fitsig**2/dummyN)
            
            if verbose: print("Reset difference of means: %.1f delta_mu"%(diff/stdofmean))
            
            if diff < 2*stdofmean: break
            
            good = np.logical_and(good, np.abs(res-mu) < clip*sig)
            mask[~good] = np.nan
            
        # Catch the last few pixels that are high flux, clip until the mean of the residual
        # of flux-smoothedFlux is close to 0
        (nr,nc) = res.shape
        y,x = np.mgrid[:nr,:nc]
        r = np.sqrt((y-nr/2)**2 + (x-nc/2)**2)
        kern = r <= 10
        kern = kern/np.sum(kern)

        for i in range(maxIter):
            sm = fftconvolve(flux*good, kern, mode='same') / fftconvolve(good, kern, mode='same')
            resid = flux-sm
            
            mu = np.nanmean(flux*mask-sm)
            sig = np.nanstd(flux*mask-sm)
            stdofmean = sig/np.sqrt(np.sum(good))
            
            if verbose: print('Residual flux mean: %.2f delta_mu'%(mu/stdofmean))
            
            if np.abs(mu) < stdofmean: break
            
            good = np.logical_and(good, np.abs(resid-mu) < clip*sig)
            mask[~good] = np.nan
        
        # Assume reference pixels are good
        mask[:,:4] = 1
        mask[:,-4:] = 1
        if self.YFRM == 0: 
            mask[:4,:] = 1
        if self.YFRM + self.YFRAMESZ == 2048:
            mask[-4:,:] = 1
        
        self.setPixMask(mask, prefix)
            
    
    def setPixMask(self, pixMask, prefix=None):
        '''
        Set the pixel mask to the one that is supplied (should have 1 for good
        pixels, nan for bad pxiels).

        The supplied mask is for the full frame.

        This routine internally determines a pixel mask for the window if
        there is one.

        This routine applies the mask to the following data cubes within
        the object if they exist:
            ff
            ffDiffs
            win

        If prefix != None, the masks are written to files:
        [prefix]_pixMask
        [prefix]_winPixMask
        '''

        if (pixMask.shape[0] != self.YFRAMESZ) or (pixMask.shape[1] != self.XFRAMESZ):
            raise Exception( "Supplied mask dimensions do not match full-frame data")

        if self.pixMask is None:
            self.pixMask = pixMask
        else:
            self.pixMask = self.pixMask * pixMask
        
        if prefix is not None:
            writeFits( self.pixMask, prefix+'_pixMask.fits', overwrite=True)

        # TODO: this still uses the old setup where only 1 window is possible
        # if self.haveWin:
        #     self.winPixMask = pixMask[self.YWINSTRT:self.YWINSTOP+1, self.XWINSTRT:self.XWINSTOP+1]

        #     if prefix is not None:
        #         writeFits( self.winPixMask, prefix+'_winPixMask.fits', overwrite=True)
        
        # Apply masks to data cubes
        if self.ff is not None:
            for read in range(self.ff.shape[0]):
                self.ff[read,:,:] = self.ff[read,:,:] * self.pixMask

        if self.ffDiffs is not None:
            for read in range(self.ffDiffs.shape[0]):
                self.ffDiffs[read,:,:] = self.ffDiffs[read,:,:] * self.pixMask

        # if self.win is not None:
        #     for win in range(self.win.shape[0]):
        #         for read in range(self.win.shape[1]):
        #             self.win[win,read,:,:] = self.win[win,read,:,:] * self.winPixMask

        # modeled flux arrays
        if self.coeffMaps is not None:
            for i in range(self.coeffMaps.shape[0]):
                self.coeffMaps[i, :, :] *= self.pixMask

        if self.model is not None:
            for read in range(self.model.shape[0]):
                self.model[read,:,:] = self.model[read,:,:] * self.pixMask
        
        if self.residual is not None:
            for read in range(self.residual.shape[0]):
                self.residual[read,:,:] = self.residual[read,:,:] * self.pixMask
        
        if self.std is not None:
            self.std = self.std * self.pixMask
            
        if self.flux is not None:
            self.flux = self.flux * self.pixMask
            
        if self.fluxErr is not None:
            self.fluxErr = self.fluxErr * self.pixMask

        # modeled flux win arrays
        # if self.winCoeffMaps is not None:
        #     for win in range(self.NWINS):
        #         for ramp in range(len(self.winCoeffMaps[win])):
        #             for i in range(self.winCoeffMaps[win][ramp].shape[0]):
        #                 self.winCoeffMaps[win][ramp][i, :, :] *= self.winPixMask
        #         self.winStd[win] = self.winStd[win] * self.winPixMask
        #         for read in range(self.winModel[win].shape[0]):
        #             self.winModel[win][read,:,:] = self.winModel[win][read,:,:] * self.winPixMask
        #             self.winResidual[win][read,:,:] = self.winResidual[win][read,:,:] * self.winPixMask
        
        # # smoothed flux and residual win arrays
        # if self.winSmooth is not None:
        #     for win in range(self.NWINS):
        #         self.winSmooth[win] = self.winSmooth[win] * self.winPixMask
        #         self.winSmoothResidual[win] = self.winSmoothResidual[win] * self.winPixMask


    def applyRefMap(self, refMap):
        for read in range(self.NFFREADS):
            for row in range(self.YFRAMESZ):
                self.data[read,row,:self.XFRAMESZ] = self.data[read,row,:self.XFRAMESZ] - refMap[read,row,:]

                if self.haveWin:
                    self.data[read,row,self.XFRAMESZ:] = self.data[read,row,self.XFRAMESZ:] - refMap[read,row,0]