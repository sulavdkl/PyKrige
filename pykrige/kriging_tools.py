from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__doc__ = """
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Methods for reading/writing ASCII grid files.

Copyright (c) 2015-2018, PyKrige Developers
"""

import numpy as np
import warnings
import io
from string import ascii_uppercase
import itertools

def write_tough_grid(x, y, z, style=1):
    """Writes gridded data to TOUGH+Hydrate mesh file (MESH). This is useful for
    exporting data to TOUGH+Hydrate program.
    
    Author:Sulav Dhakal
    Organization: Louisiana State University

    Parameters
    ----------
    x : array_like, shape (N,) or (N, 1)
        X-coordinates of grid points at center of cells.
    y : array_like, shape (M,) or (M, 1)
        Y-coordinates of grid points at center of cells.
    z : array_like, shape (M, N)
        Gridded data values. May be a masked array.
    filename : string, optional
        Name of output *.asc file. Default name is 'MESH'.
    style : int, optional
        Determines how to write the *.asc file header.
        Specifying 1 writes out DX, DY, XLLCENTER, YLLCENTER.
        Specifying 2 writes out CELLSIZE (note DX must be the same as DY),
        XLLCORNER, YLLCORNER. Default is 1.
    """
    filename='MESH'
    filename2='INCON'

    if np.ma.is_masked(z):
        z = np.array(z.tolist(-999.))

    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    z = np.squeeze(np.array(z))
    nrows = z.shape[0]
    ncols = z.shape[1]
    
    nr = 0
    def iter_all_strings():
        for size in itertools.count(1):
            for s in itertools.product(ascii_uppercase, repeat=size):
                yield "".join(s)

    
    if z.ndim != 2:
        raise ValueError("Two-dimensional grid is required to "
                         "write *.asc grid.")
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Dimensions of X and/or Y coordinate arrays are not "
                         "as expected. Could not write *.asc grid.")
    if z.shape != (y.size, x.size):
        warnings.warn("Grid dimensions are not as expected. "
                      "Incorrect *.asc file generation may result.",
                      RuntimeWarning)
    if np.amin(x) != x[0] or np.amin(y) != y[0]:
        warnings.warn("Order of X or Y coordinates is not as expected. "
                      "Incorrect *.asc file generation may result.",
                      RuntimeWarning)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    if abs((x[-1] - x[0])/(x.shape[0] - 1)) != dx or \
       abs((y[-1] - y[0])/(y.shape[0] - 1)) != dy:
        raise ValueError("X or Y spacing is not constant; *.asc "
                         "grid cannot be written.")
    cellsize = -1
    if style == 2:
        if dx != dy:
            raise ValueError("X and Y spacing is not the same. "
                             "Cannot write *.asc file in the specified format.")
        cellsize = dx

    xllcenter = x[0]
    yllcenter = y[0]

    # Note that these values are flagged as -1. If there is a problem in trying
    # to write out style 2, the -1 value will appear in the output file.
    xllcorner = -1
    yllcorner = -1
    if style == 2:
        xllcorner = xllcenter - dx/2.0
        yllcorner = yllcenter - dy/2.0

    no_data = -999.
    elements = []

    nn = nrows*ncols
    elements = range(1,nn+1,1)
    elx = np.tile(x,nrows)
    ely = np.repeat(y,ncols)
    elvalues = z.flatten()
    cony = []
    conx = []
    bdy_elements =[]
    elmap = np.array(elements).reshape(np.shape(z))
    for ii in range(ncols):
        map1 = np.argwhere(z[:,ii]!=-999)
        if len(map1) == 0:
            continue
        map2 = [elmap[jj,ii] for jj in map1[:]]
        map3= [list(x) for x in map2]
        conymap= sum(map3, [])
        cony2=[]
        for first, second in zip(conymap, conymap[1:]):
            elvol = dx*dy
            dist1 = dy/2
            dist2 = dy/2
            pair = np.array([first, second, dist1, dist2, elvol])
            cony2 = np.append(cony2, pair)
            cony2= cony2.reshape(-1,5)
        cony2[0,2] = 0.001
        cony2[-1,2] = 0.001
        bdy_elements= np.append(bdy_elements, int(cony2[0,0]), int(cony2[-1,0]))
        cony= np.append(cony, cony2).reshape(-1,5)
            
    for ii in range(nrows):
        map1 = np.argwhere(z[ii,:]!=-999)
        if len(map1) == 0:
            continue
        map2 = [elmap[ii,jj] for jj in map1[:]]
        map3= [list(x) for x in map2]
        conxmap= sum(map3, [])
        
        for first, second in zip(conxmap, conxmap[1:]):
            elvol = dx*dy
            dist1 = dx/2
            dist2 = dx/2
            pair = np.array([first, second, dist1, dist2, elvol])
            
            conx = np.append(conx, pair)
        conx= conx.reshape(-1,5)

    elvol= np.repeat(dx*dy,nn)
    poro = elvalues*0.005
    perm = poro*1e-10
    el_array = np.column_stack((elements, elx, ely, elvalues, elvol, poro, perm))
    active_elements = el_array[el_array[:,3]!=-999]
    
    with io.open(filename, 'w') as f:
        if style == 1:
            f.write("ELEME----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8"+"\n")
            # f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            
            # f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            
            
            # f.write("XLLCENTER      " + '{:<10.2f}'.format(xllcenter) + '\n')
            # f.write("YLLCENTER      " + '{:<10.2f}'.format(yllcenter) + '\n')
            # f.write("DX             " + '{:<10.2f}'.format(dx) + '\n')
            # f.write("DY             " + '{:<10.2f}'.format(dy) + '\n')
            # f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
            for ii in range(len(active_elements)):
                eleme_string = '{0:5d}{1:^10}{2:5s}{3:10.4f}{4:^20}{5:10.4f}{6:10.4f}{7:10.4f}\n' \
                    .format(int(active_elements[ii,0]), ' ','sand1',active_elements[ii,4],' ', active_elements[ii,1], 10, -1*active_elements[ii,2]) 
                f.write(eleme_string)
            f.write('\n'+'CONNE----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'+'\n')

            # conx = [item for item in elements[:] if item % ncols != 0]

            # conx2 =[item+1 for item in conx]
            
            # cony = elements[:-(ncols-1)]
            # cony2=[item+ncols for item in cony]
            # result1 = all(elem in active_elements[:,0]  for elem in cony)
            # result2 = all(elem in active_elements[:,0]  for elem in cony)
            
            # cony3 = list(set(active_elements[:,0]).intersection(set(list(cony))))
            # cony4 = list(set(active_elements[:,0]).intersection(set(cony2)))


            # with io.open('connections.txt', 'w') as f:
            for ii in range(len(conx)):
                connex_string ='{0:5d}{1:5d}{2:^19}{3:1d}{4:10.4f}{5:10.4f}{6:10.4f}{7:10.4f}\n'.format(int(conx[ii,0]),int(conx[ii,1]), ' ',1,dx,dx, elvol[ii],0)
                f.write(connex_string)
            for ii in range(len(cony)):
                conney_string ='{0:5d}{1:5d}{2:^19}{3:1d}{4:10.4f}{5:10.4f}{6:10.4f}{7:10.4f}\n'.format(int(cony[ii,0]),int(cony2[ii,1]), ' ',3,dy,dy, elvol[ii],1)
                f.write(conney_string)
            f.write('\n')
        elif style == 2:
            f.write("ELEME----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8"+"\n")
            # f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            
            # f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            
            
            # f.write("XLLCENTER      " + '{:<10.2f}'.format(xllcenter) + '\n')
            # f.write("YLLCENTER      " + '{:<10.2f}'.format(yllcenter) + '\n')
            # f.write("DX             " + '{:<10.2f}'.format(dx) + '\n')
            # f.write("DY             " + '{:<10.2f}'.format(dy) + '\n')
            # f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
            for ii in range(len(elements)):
                eleme_string = '{0:8d}{1:^10}{2:5s}{3:10.4f}{4:^20}{5:10.4f}{6:10.4f}{7:10.4f}\n' \
                    .format(elements[ii], ' ','sand1',elvol,' ', elx[ii], 10, ely[ii]) 
            
                f.write(eleme_string)
            f.write('\n'+'CONNE----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8'+'\n')

        else:
            raise ValueError("style kwarg must be either 1 or 2.")

        # for m in range(z.shape[0] - 1, -1, -1):
        #     for n in range(z.shape[1]):
        #         f.write('{:<16.2f}'.format(z[m, n]))
        #     if m != 0:
        #         f.write('\n')
        
        
        
        f.close()
        with io.open(filename2, 'w') as f:
            f.write("INCON  Initial conditions for    {:d} elements at time".format(len(active_elements))+"\n")
            # f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            
            # f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            
            
            # f.write("XLLCENTER      " + '{:<10.2f}'.format(xllcenter) + '\n')
            # f.write("YLLCENTER      " + '{:<10.2f}'.format(yllcenter) + '\n')
            # f.write("DX             " + '{:<10.2f}'.format(dx) + '\n')
            # f.write("DY             " + '{:<10.2f}'.format(dy) + '\n')
            # f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
            for ii in range(len(active_elements)):
                incon_string1 = '{0:5d}{1:^10}{2:15.8f}{3:s}{4:s}{5:^35}{6:15.8e}{7:15.8e}{8:15.8e}\n' \
                    .format(int(active_elements[ii,0]), ' ',active_elements[ii,5],'  ','Aqu',' ', active_elements[ii,6], active_elements[ii,6], 0.1*active_elements[ii,6]) 
                f.write(incon_string1)
                p1 = 1e5*active_elements[ii,2]
                p2 = 2e-4
                p3 = 3.5e-2
                p4 = 3+0.03*(active_elements[ii,2]-1900)
                incon_string2 = "{0:20.13e}{1:20.13e}{2:20.13e}{3:20.13e}\n".format(p1, p2, p3, p4)
                f.write(incon_string2)
                f.write('\n')
            



        
        

def write_asc_grid(x, y, z, filename='output.asc', style=1):
    """Writes gridded data to ASCII grid file (*.asc). This is useful for
    exporting data to a GIS program.

    Parameters
    ----------
    x : array_like, shape (N,) or (N, 1)
        X-coordinates of grid points at center of cells.
    y : array_like, shape (M,) or (M, 1)
        Y-coordinates of grid points at center of cells.
    z : array_like, shape (M, N)
        Gridded data values. May be a masked array.
    filename : string, optional
        Name of output *.asc file. Default name is 'output.asc'.
    style : int, optional
        Determines how to write the *.asc file header.
        Specifying 1 writes out DX, DY, XLLCENTER, YLLCENTER.
        Specifying 2 writes out CELLSIZE (note DX must be the same as DY),
        XLLCORNER, YLLCORNER. Default is 1.
    """

    if np.ma.is_masked(z):
        z = np.array(z.tolist(-999.))

    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    z = np.squeeze(np.array(z))
    nrows = z.shape[0]
    ncols = z.shape[1]

    if z.ndim != 2:
        raise ValueError("Two-dimensional grid is required to "
                         "write *.asc grid.")
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("Dimensions of X and/or Y coordinate arrays are not "
                         "as expected. Could not write *.asc grid.")
    if z.shape != (y.size, x.size):
        warnings.warn("Grid dimensions are not as expected. "
                      "Incorrect *.asc file generation may result.",
                      RuntimeWarning)
    if np.amin(x) != x[0] or np.amin(y) != y[0]:
        warnings.warn("Order of X or Y coordinates is not as expected. "
                      "Incorrect *.asc file generation may result.",
                      RuntimeWarning)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    if abs((x[-1] - x[0])/(x.shape[0] - 1)) != dx or \
       abs((y[-1] - y[0])/(y.shape[0] - 1)) != dy:
        raise ValueError("X or Y spacing is not constant; *.asc "
                         "grid cannot be written.")
    cellsize = -1
    if style == 2:
        if dx != dy:
            raise ValueError("X and Y spacing is not the same. "
                             "Cannot write *.asc file in the specified format.")
        cellsize = dx

    xllcenter = x[0]
    yllcenter = y[0]

    # Note that these values are flagged as -1. If there is a problem in trying
    # to write out style 2, the -1 value will appear in the output file.
    xllcorner = -1
    yllcorner = -1
    if style == 2:
        xllcorner = xllcenter - dx/2.0
        yllcorner = yllcenter - dy/2.0

    no_data = -999.

    with io.open(filename, 'w') as f:
        if style == 1:
            f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            f.write("XLLCENTER      " + '{:<10.2f}'.format(xllcenter) + '\n')
            f.write("YLLCENTER      " + '{:<10.2f}'.format(yllcenter) + '\n')
            f.write("DX             " + '{:<10.2f}'.format(dx) + '\n')
            f.write("DY             " + '{:<10.2f}'.format(dy) + '\n')
            f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
        elif style == 2:
            f.write("NCOLS          " + '{:<10n}'.format(ncols) + '\n')
            f.write("NROWS          " + '{:<10n}'.format(nrows) + '\n')
            f.write("XLLCORNER      " + '{:<10.2f}'.format(xllcorner) + '\n')
            f.write("YLLCORNER      " + '{:<10.2f}'.format(yllcorner) + '\n')
            f.write("CELLSIZE       " + '{:<10.2f}'.format(cellsize) + '\n')
            f.write("NODATA_VALUE   " + '{:<10.2f}'.format(no_data) + '\n')
        else:
            raise ValueError("style kwarg must be either 1 or 2.")

        for m in range(z.shape[0] - 1, -1, -1):
            for n in range(z.shape[1]):
                f.write('{:<16.2f}'.format(z[m, n]))
            if m != 0:
                f.write('\n')


def read_asc_grid(filename, footer=0):
    """Reads ASCII grid file (*.asc).

    Parameters
    ----------
    filename : str
        Name of *.asc file.
    footer : int, optional
        Number of lines at bottom of *.asc file to skip.

    Returns
    -------
    grid_array : numpy array, shape (M, N)
        (M, N) array of grid values, where M is number of Y-coordinates and
        N is number of X-coordinates. The array entry corresponding to
        the lower-left coordinates is at index [M, 0], so that
        the array is oriented as it would be in X-Y space.
    x : numpy array, shape (N,)
        1D array of N X-coordinates.
    y : numpy array, shape (M,)
        1D array of M Y-coordinates.
    CELLSIZE : tuple or float
        Either a two-tuple of (x-cell size, y-cell size),
        or a float that specifies the uniform cell size.
    NODATA : float
        Value that specifies which entries are not actual data.
    """

    ncols = None
    nrows = None
    xllcorner = None
    xllcenter = None
    yllcorner = None
    yllcenter = None
    cellsize = None
    dx = None
    dy = None
    no_data = None
    header_lines = 0
    with io.open(filename, 'r') as f:
        while True:
            string, value = f.readline().split()
            header_lines += 1
            if string.lower() == 'ncols':
                ncols = int(value)
            elif string.lower() == 'nrows':
                nrows = int(value)
            elif string.lower() == 'xllcorner':
                xllcorner = float(value)
            elif string.lower() == 'xllcenter':
                xllcenter = float(value)
            elif string.lower() == 'yllcorner':
                yllcorner = float(value)
            elif string.lower() == 'yllcenter':
                yllcenter = float(value)
            elif string.lower() == 'cellsize':
                cellsize = float(value)
            elif string.lower() == 'cell_size':
                cellsize = float(value)
            elif string.lower() == 'dx':
                dx = float(value)
            elif string.lower() == 'dy':
                dy = float(value)
            elif string.lower() == 'nodata_value':
                no_data = float(value)
            elif string.lower() == 'nodatavalue':
                no_data = float(value)
            else:
                raise IOError("could not read *.asc file. Error in header.")

            if (ncols is not None) and \
               (nrows is not None) and \
               (((xllcorner is not None) and (yllcorner is not None)) or
                ((xllcenter is not None) and (yllcenter is not None))) and \
               ((cellsize is not None) or ((dx is not None) and (dy is not None))) and \
               (no_data is not None):
                break

    raw_grid_array = np.genfromtxt(filename, skip_header=header_lines,
                                   skip_footer=footer)
    grid_array = np.flipud(raw_grid_array)

    if nrows != grid_array.shape[0] or ncols != grid_array.shape[1]:
        raise IOError("Error reading *.asc file. Encountered problem "
                      "with header: NCOLS and/or NROWS does not match "
                      "number of columns/rows in data file body.")

    if xllcorner is not None and yllcorner is not None:
        if dx is not None and dy is not None:
            xllcenter = xllcorner + dx/2.0
            yllcenter = yllcorner + dy/2.0
        else:
            xllcenter = xllcorner + cellsize/2.0
            yllcenter = yllcorner + cellsize/2.0

    if dx is not None and dy is not None:
        x = np.arange(xllcenter, xllcenter + ncols*dx, dx)
        y = np.arange(yllcenter, yllcenter + nrows*dy, dy)
    else:
        x = np.arange(xllcenter, xllcenter + ncols*cellsize, cellsize)
        y = np.arange(yllcenter, yllcenter + nrows*cellsize, cellsize)

    # Sometimes x and y and can be an entry too long due to imprecision
    # in calculating the upper cutoff for np.arange(); this bit takes care of
    # that potential problem.
    if x.size == ncols + 1:
        x = x[:-1]
    if y.size == nrows + 1:
        y = y[:-1]

    if cellsize is None:
        cellsize = (dx, dy)

    return grid_array, x, y, cellsize, no_data
