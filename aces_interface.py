
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-ACES-INTERFACE-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-


def read_eq_geom_aces(filename):

    '''
    General function to read the "eq_geom" file produced
    by ACES calculation (with fcm keyword). 

    The following data are read:

    1. natom: number of atoms
    2. amass: atomic masses
    3. cgeom: cartesian equilibrium geometry

    Pattern:
    - Line #2: Number of atoms
    - Line #4: Atomic masses 
    - Line #6 onwards: Cartesian coordinates (order follows from atomic masses)
    '''
    
    # Read file
    try:
	with open(filename) as f:
	    get_all_lines_list = f.readlines()
    except EnvironmentError:
	print 'Something not right with EQ_GEOM file.'
	print 'IOError / OSError / WindowsError. Check. Fix. Rerun.'
    

    # Process data (Each line is a string in the list, strip and split are game)
    natom = int(get_all_lines_list[1].strip().split()[0])
    amass = np.array([float(item) for item in get_all_lines_list[3].strip().split()]) 

    # Little more work for cartesian geometry
    cgeom = np.zeros((natom, 3))
    for i in range(natom):
	cgeom[i,:] = np.array([float(item) for item in get_all_lines_list[i+5].strip().split()])


    return natom, amass, cgeom


def read_normal_fdif_aces(filename, natom):

    '''
    General function to read the "normal_fdif" file produced
    by ACES calculation (with fcm keyword).

    The following data are read:

    1. xnormal: mass weighted cartesian normal coordinates (3N, 3N-6)
    2. freq: harmonic vibrational frequencies (3N-6)
    3. irreps: irreducible representation for each normal mode [3N-6]
    4. mtype: a list of characters 'r' or 'i' indicating whether a normal mode is real or imaginary

    '''
    
    nmode = 3*natom - 3

    # Read file
    try:
	with open(filename) as f:
	    get_all_lines_list = f.readlines()
    except EnvironmentError:
	print 'Something not right with NORMAL_FDIF file.'
	print 'IOError / OSError / WindowsError. Check. Fix. Rerun.'

    
    #Initialize
    mtype = ['']*nmode
    irreps = ['']*nmode
    freq   = np.zeros(nmode)
    xnormal = np.zeros((3*natom, nmode))
    
    #1. First loop for frequencies and irreps (i+natom)
    for i in range(nmode):
	irreps[i] = get_all_lines_list[i*(1+natom)].strip().split()[0] 
	mtype[i]  = get_all_lines_list[i*(1+natom)].strip().split()[3]
	freq[i] = float(get_all_lines_list[i*(1+natom)].strip().split()[2])

    #2. Second loop for normal coordinates
    k=1
    for i in range(nmode):
	m=0
	for j in range(natom):	
	    temp = get_all_lines_list[k+j].strip().split()
	    for l in range(3):
		xnormal[m][i] = float(temp[l+1])
		m=m+1
	k=(k+1)+natom

    return xnormal, freq, irreps, mtype


def read_fcmfinal_aces(filename):

    '''
    Read Hessian in Cartesian coordinates (3N, 3N)
    '''

    # Read file
    try:
	with open(filename) as f:
	    get_all_lines_list = f.readlines()
    except EnvironmentError:
	print 'Something not right with FCMFINAL file.'
	print 'IOError / OSError / WindowsError. Check. Fix. Rerun.'

    # Process data (Each line is a string in the list, strip and split are game)
    natom = int(get_all_lines_list[0].strip().split()[0])

    # Initialize
    force_constant_matrix = np.zeros((3*natom, 3*natom))

    # Use np.loadtxt to directly read the columns
    hessian_data = np.loadtxt(filename, usecols=(0,1,2), skiprows=1, unpack=True)
    force_constant_matrix = np.reshape(hessian_data.T, (3*natom, 3*natom))

    return force_constant_matrix


# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-END-ACES-INTERFACE-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-


