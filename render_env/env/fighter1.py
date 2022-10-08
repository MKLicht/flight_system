import math
from math import sin, cos, tan

import numpy as np

PLA_TURN_RATE = 0.1
PLA_LIFT_RATE = 20.0
PLA_ROLL_RATE = 30.0


class Fighter:
    def __init__(self):
        self.old = []
        self.sta = []

    def __fix(self, dinput):
        if (dinput >= 0.0):
            dout = math.floor(dinput)
        elif (dinput < 0.0):
            dout = math.ceil(dinput)
        return dout

    def __sign(self, dinput):
        if (dinput > 0.0):
            dout = 1
        elif (dinput < 0.0):
            dout = -1
        elif (dinput == 0.0):
            dout = 0
        return dout

    #########damping aero-coeffs###########
    def __damping(self, alpha, coeff):
        A = np.array([
            [-.267, .110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.5, 1.49, 1.83, 1.21],
            [.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04],
            [-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -2.27],
            [-8.8, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29, -29.8, -38.3, -35.3],
            [-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .1, .447, -.330],
            [-.36, -.359, -.443, -.42, -.383, -.375, -.329, -.294, -.23, -.21, -.12, -.1],
            [-7.21, -.54, -5.23, -5.26, -6.11, -6.64, -5.69, -6, -6.2, -6.4, -6.6, -6],
            [-.38, -.363, -.378, -.386, -.37, -.453, -.55, -.582, -.595, -.637, -1.02, -.84],
            [.061, .052, .052, -.012, -.013, -.024, .05, .15, .13, .158, .24, .15]])

        s = .2 * alpha
        k = self.__fix(s)
        if (k <= -2):
            k = -1
        elif (k >= 9):
            k = 8
        da = s - k
        L = k + self.__fix(1.1 * self.__sign(da))
        k = k + 3
        L = L + 3

        coeff[0] = A[0][k - 1] + math.fabs(da) * (A[0][L - 1] - A[0][k - 1])
        coeff[1] = A[1][k - 1] + math.fabs(da) * (A[1][L - 1] - A[1][k - 1])
        coeff[2] = A[2][k - 1] + math.fabs(da) * (A[2][L - 1] - A[2][k - 1])
        coeff[3] = A[3][k - 1] + math.fabs(da) * (A[3][L - 1] - A[3][k - 1])
        coeff[4] = A[4][k - 1] + math.fabs(da) * (A[4][L - 1] - A[4][k - 1])
        coeff[5] = A[5][k - 1] + math.fabs(da) * (A[5][L - 1] - A[5][k - 1])
        coeff[6] = A[6][k - 1] + math.fabs(da) * (A[6][L - 1] - A[6][k - 1])
        coeff[7] = A[7][k - 1] + math.fabs(da) * (A[7][L - 1] - A[7][k - 1])
        coeff[8] = A[8][k - 1] + math.fabs(da) * (A[8][L - 1] - A[8][k - 1])

    ##########control inputs###############
    def __dmomdocon(self, alpha, beta, coeff):
        ALA = np.array([
            [-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017],
            [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016],
            [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014],
            [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012],
            [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011],
            [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010],
            [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008]])

        ALR = np.array([
            [.005, .017, .014, .010, -.005, .009, .019, .005, 0.0, -.005, -.011, .008],
            [.007, .016, .014, .014, .013, .009, .012, .005, 0.0, .004, .009, .007],
            [.013, .013, .011, .012, .011, .009, .008, .005, -.002, .005, .003, .005],
            [.018, .015, .015, .014, .014, .014, .014, .015, .013, .011, .006, .001],
            [.015, .014, .013, .013, .012, .011, .011, .010, .008, .008, .007, .003],
            [.021, .011, .010, .011, .010, .009, .008, .010, .006, .005, 0.0, .001],
            [.023, .010, .011, .011, .011, .010, .008, .010, .006, .014, .020, 0.0]])

        ANA = np.array([
            [.001, -.027, -.017, -.013, -.012, -.016, .001, .017, .011, .017, .008, .016],
            [.002, -.014, -.016, -.016, -.014, -.019, -.021, .002, .012, .016, .015, .011],
            [-.006, -.008, -.006, -.006, -.005, -.008, -.005, .007, .004, .007, .006, .006],
            [-.011, -.011, -.010, -.009, -.008, -.006, 0.0, .004, .007, .010, .004, .010],
            [-.015, -.015, -.014, -.012, -.011, -.008, -.002, .002, .006, .012, .011, .011],
            [-.024, -.010, -.004, -.002, -.001, .003, .014, .006, -.001, .004, .004, .006],
            [-.022, .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001]])

        ANR = np.array([
            [-.018, -.052, -.052, -.052, -.054, -.049, -.059, -.051, -.030, -.037, -.026, -.013],
            [-.028, -.051, -.043, -.046, -.045, -.049, -.057, -.052, -.030, -.033, -.030, -.008],
            [-.037, -.041, -.038, -.040, -.040, -.038, -.037, -.030, -.027, -.024, -.019, -.013],
            [-.048, -.045, -.045, -.045, -.044, -.045, -.047, -.048, -.049, -.045, -.033, -.016],
            [-.043, -.044, -.041, -.041, -.040, -.038, -.034, -.035, -.035, -.029, -.022, -.009],
            [-.052, -.034, -.036, -.036, -.035, -.028, -.024, -.023, -.020, -.016, -.010, -.014],
            [-.062, -.034, -.027, -.028, -.027, -.027, -.023, -.023, -.019, -.009, -.025, -.010]
        ])

        s = 0.2 * (alpha)
        k = self.__fix(s)
        if (k <= -2):
            k = -1
        elif (k >= 9):
            k = 8
        da = s - k
        L = k + self.__fix(1.1 * self.__sign(da))
        s = 0.2 * math.fabs(beta)
        m = self.__fix(s)
        # if(m >= 7):
        #    m = 6
        if (m == 0):
            m = 1
        elif (m >= 6):
            m = 5

        db = s - m
        n = m + 1
        k = k + 3
        L = L + 3
        m = m + 1
        n = n + 1

        t = ALA[m - 1][k - 1]
        u = ALA[n - 1][k - 1]
        v = t + math.fabs(da) * (ALA[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (ALA[n - 1][L - 1] - u)
        dlda = v + (w - v) * db

        t = ALR[m - 1][k - 1]
        u = ALR[n - 1][k - 1]
        v = t + math.fabs(da) * (ALR[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (ALR[n - 1][L - 1] - u)
        dldr = v + (w - v) * db

        t = ANA[m - 1][k - 1]
        u = ANA[n - 1][k - 1]
        v = t + math.fabs(da) * (ANA[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (ANA[n - 1][L - 1] - u)
        dnda = v + (w - v) * db

        t = ANR[m - 1][k - 1]
        u = ANR[n - 1][k - 1]
        v = t + math.fabs(da) * (ANR[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (ANR[n - 1][L - 1] - u)
        dndr = v + (w - v) * db

        coeff[0] = dlda
        coeff[1] = dldr
        coeff[2] = dnda
        coeff[3] = dndr

    ##########Cl and Cn aero-coeff#########
    def __clcn(self, alpha, beta, coeff):
        AL = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-.001, -.004, -.008, -.012, -.016, -.019, -.020, -.020, -.015, -.008, -.013, -.015],
            [-.003, -.009, -.017, -.024, -.030, -.034, -.040, -.037, -.016, -.002, -.010, -.019],
            [-.001, -.010, -.020, -.030, -.039, -.044, -.050, -.049, -.023, -.006, -.014, -.027],
            [0, -.010, -.022, -.034, -.047, -.046, -.059, -.061, -.033, -.036, -.035, -.035],
            [.007, -.010, -.023, -.034, -.049, -.046, -.068, -.071, -.060, -.058, -.062, -.059],
            [.009, -.011, -.023, -.037, -.050, -.047, -.074, -.079, -.091, -.076, -.077, -.076]
        ])

        AN = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [.018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033],
            [.038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057],
            [.056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073],
            [.064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041],
            [.074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013],
            [.079, .090, .106, .106, .096, .080, .068, .030, .064, .015, .011, -.001]
        ])

        s = .2 * (alpha)
        k = self.__fix(s)
        if (k <= -2):
            k = -1
        elif (k >= 9):
            k = 8
        da = s - k
        L = k + self.__fix(1.1 * self.__sign(da))
        s = .2 * math.fabs(beta)
        m = self.__fix(s)
        if (m == 0):
            m = 1
        elif (m >= 6):
            m = 5
        db = s - m
        n = m + self.__fix(1.1 * self.__sign(db))
        k = k + 3
        L = L + 3
        m = m + 1
        n = n + 1

        t = AL[m - 1][k - 1]
        u = AL[n - 1][k - 1]
        v = t + math.fabs(da) * (AL[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (AL[n - 1][L - 1] - u)
        dum = v + (w - v) * math.fabs(db)
        cl = dum * self.__sign(beta)

        t = AN[m - 1][k - 1]
        u = AN[n - 1][k - 1]
        v = t + math.fabs(da) * (AN[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (AN[n - 1][L - 1] - u)
        dum = v + (w - v) * math.fabs(db)
        cn = dum * self.__sign(beta)

        coeff[0] = cl
        coeff[1] = cn

    ##########Cx and Cm aero-coeffs########
    def __cxcm(self, alpha, dele, coeff):
        AX = np.array([
            [-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166],
            [-.048, -.038, -.040, -.021, .016, .083, .127, .137, .162, .177, .179, .167],
            [-.022, -.020, -.021, -.004, .032, .094, .128, .130, .154, .161, .155, .138],
            [-.040, -.038, -.039, -.025, .006, .062, .087, .085, .100, .110, .104, .091],
            [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040]])

        AM = np.array([
            [.205, .168, .186, .196, .213, .251, .245, .238, .252, .231, .198, .192],
            [.081, .077, .107, .110, .110, .141, .127, .119, .133, .108, .081, .093],
            [-.046, -.020, -.009, -.005, -.006, .010, .006, -.001, .014, 0.0, -.013, .032],
            [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006],
            [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005]])

        s = .2 * (alpha)
        k = self.__fix(s)
        if (k <= -2):
            k = -1
        elif (k >= 9):
            k = 8
        da = s - k
        L = k + self.__fix(1.1 * self.__sign(da))
        s = dele / 12.0
        m = self.__fix(s)
        if (m <= -2):
            m = -1
        elif (m >= 2):
            m = 1
        de = s - m
        n = m + self.__fix(1.1 * self.__sign(de))
        k = k + 3
        L = L + 3
        m = m + 3
        n = n + 3

        t = AX[m - 1][k - 1]
        u = AX[n - 1][k - 1]
        v = t + math.fabs(da) * (AX[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (AX[n - 1][L - 1] - u)
        cx = v + (w - v) * math.fabs(de)

        t = AM[m - 1][k - 1]
        u = AM[n - 1][k - 1]
        v = t + math.fabs(da) * (AM[m - 1][L - 1] - t)
        w = u + math.fabs(da) * (AM[n - 1][L - 1] - u)
        cm = v + (w - v) * math.fabs(de)

        coeff[0] = cx
        coeff[1] = cm

    ##########Cz aero-coeff################
    def __cz(self, alpha, beta, dele, coeff):
        A = np.array([.770, .241, -.100, -.416, -.731, -1.053, -1.366,
                      -1.646, -1.917, -2.120, -2.248, -2.229])
        s = .2 * (alpha)
        k = self.__fix(s)
        if (k <= -2):
            k = -1
        elif (k >= 9):
            k = 8
        da = s - k
        L = k + self.__fix(1.1 * self.__sign(da))

        k = k + 3
        L = L + 3

        s = A[k - 1] + math.fabs(da) * (A[L - 1] - A[k - 1])
        cz = s * (1 - math.pow((beta / 57.3), 2)) - .19 * (dele) / 25
        coeff[0] = cz

    ##########atmospheric effect###########
    def __atmos(self, alt, vt, coeff):
        rho0 = 2.377e-3
        tfac = 1 - .703e-5 * (alt)
        if (tfac <= 0):
            print("Warning: The aircraft overflew the altitude limit")
            tfac = 0.75144
        temp = 519.0 * tfac
        if (alt >= 35000.0):
            temp = 390
        rho = rho0 * pow(tfac, 4.14)
        mach = (vt) / math.sqrt(1.4 * 1716.3 * temp)
        qbar = .5 * rho * pow(vt, 2)
        ps = 1715.0 * rho * temp
        if (ps == 0):
            ps = 1715

        coeff[0] = mach
        coeff[1] = qbar
        coeff[2] = ps

    def __warp(self, h, p, r):
        p %= 360
        h %= 360
        r %= 360
        if (90 < p <= 270):
            h = (h + 180) % 360
            r = (r + 180) % 360
            p = 180 - p 
        if (180 < p <=360):
            p = -(360 - p)
        if (180 < h <=360):
            h = -(360 - h)
        if (180 < r <=360):
            r = -(360 - r)
        return h, p, r

    ##########flight dynamics##############
    def __dynamics(self, pre_state, action, dt):
        # F16 Nasa Data
        g = 9.8  # gravity
        m = 9295.44  # mass
        B = 9.144  # span
        S = 27.87  # planform area
        cbar = 3.45  # mean aero chord
        xcgr = 0.35
        xcg = 0.30

        Heng = 0.0
        pi = math.acos(-1)
        r2d = 180.0 / pi

        Jy = 75674
        Jxz = 1331
        Jz = 85552
        Jx = 12875

        x, y, z = pre_state[1], pre_state[0], pre_state[2]
        heading, pitch, roll = pre_state[3], pre_state[4], pre_state[5]
        V, U, W = pre_state[6], pre_state[7], pre_state[8]
        vt = pre_state[18]
        P, Q, R = pre_state[9], pre_state[10], pre_state[11]
        alpha, beta = pre_state[12], pre_state[13]
        T, ail, el, rud = pre_state[14], pre_state[15], pre_state[16], pre_state[17]

        ac_v, ac_p, ac_r, ac_y = action[0], action[1], action[2], action[3]

        if (ac_v == 1):
            T = 80000
        elif (ac_v == -1):
            T = -20000
        if (ac_p == 1):
            Q, el = 0.02*dt, 30*dt
        elif (ac_p == -1):
            Q, el = -0.02*dt, -30*dt
        if (ac_r == 1):
            P, ail = 0.02*dt, 21.5*dt
        elif (ac_r == -1):
            P, ail = -0.02*dt, -21.5*dt
        if (ac_y == 1):
            R, rud = 0.001*dt, -10.0*dt
        elif (ac_y == -1):
            R, rud = -0.001*dt, 10.0*dt

        alt = z * 100.0

        # normalized aganist max angle
        dail = ail / 21.5
        drud = rud / 30.0

        ######atmospheric effects####
        temp = [0, 0, 0]
        self.__atmos(alt, vt, temp)
        mach = temp[0]
        qbar = temp[1]
        ps = temp[2]

        ############AeroData###############
        dlef = 0.0
        temp1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.__damping(alpha, temp1)
        Cxq = temp1[0]
        Cyr = temp1[1]
        Cyp = temp1[2]
        Czq = temp1[3]
        Clr = temp1[4]
        Clp = temp1[5]
        Cmq = temp1[6]
        Cnr = temp1[7]
        Cnp = temp1[8]

        temp2 = [0, 0, 0, 0]
        self.__dmomdocon(alpha, beta, temp2)
        delta_Cl_a20 = temp2[0]
        delta_Cl_r30 = temp2[1]
        delta_Cn_a20 = temp2[2]
        delta_Cn_r30 = temp2[3]

        temp3 = [0, 0]
        self.__clcn(alpha, beta, temp3)
        Cl = temp[0]
        Cn = temp[1]

        temp4 = [0, 0]
        self.__cxcm(alpha, el, temp4)
        Cx = temp4[0]
        Cm = temp4[1]

        temp5 = [0]
        self.__cz(alpha, beta, el, temp5)
        Cz = temp5[0]

        Cy = -.02 * beta + .021 * dail + .086 * drud

        delta_Cx_lef = 0.0
        delta_Cz_lef = 0.0
        delta_Cm_lef = 0.0
        delta_Cy_lef = 0.0
        delta_Cn_lef = 0.0
        delta_Cl_lef = 0.0
        delta_Cxq_lef = 0.0
        delta_Cyr_lef = 0.0
        delta_Cyp_lef = 0.0
        delta_Czq_lef = 0.0
        delta_Clr_lef = 0.0
        delta_Clp_lef = 0.0
        delta_Cmq_lef = 0.0
        delta_Cnr_lef = 0.0
        delta_Cnp_lef = 0.0
        delta_Cy_r30 = 0.0
        delta_Cy_a20 = 0.0
        delta_Cy_a20_lef = 0.0
        delta_Cn_a20_lef = 0.0
        delta_Cl_a20_lef = 0.0
        delta_Cnbeta = 0.0
        delta_Clbeta = 0.0
        delta_Cm = 0.0
        eta_el = 1.0
        delta_Cm_ds = 0.0

        ########C_tot###############
        dXdQ = (cbar / (2 * vt)) * (Cxq + delta_Cxq_lef * dlef)
        Cx_tot = Cx + delta_Cx_lef * dlef + dXdQ * Q

        dZdQ = (cbar / (2 * vt)) * (Czq + delta_Cz_lef * dlef)
        Cz_tot = Cz + delta_Cz_lef * dlef + dZdQ * Q

        dMdQ = (cbar / (2 * vt)) * (Cmq + delta_Cmq_lef * dlef)
        Cm_tot = Cm * eta_el + Cz_tot * (xcgr - xcg) + delta_Cm_lef * dlef + dMdQ * Q + delta_Cm + delta_Cm_ds

        dYdail = delta_Cy_a20 + delta_Cy_a20_lef * dlef
        dYdR = (B / (2 * vt)) * (Cyr + delta_Cyr_lef * dlef)
        dYdP = (B / (2 * vt)) * (Cyp + delta_Cyp_lef * dlef)
        Cy_tot = Cy + delta_Cy_lef * dlef + dYdail * dail + delta_Cy_r30 * drud + dYdR * R + dYdP * P

        dNdail = delta_Cn_a20 + delta_Cn_a20_lef * dlef
        dNdR = (B / (2 * vt)) * (Cnr + delta_Cnr_lef * dlef)
        dNdP = (B / (2 * vt)) * (Cnp + delta_Cnp_lef * dlef)
        Cn_tot = Cn + delta_Cn_lef * dlef - Cy_tot * (xcgr - xcg) * (
                cbar / B) + dNdail * dail + delta_Cn_r30 * drud + dNdR * R + dNdP * P + delta_Cnbeta * beta

        dLdail = delta_Cl_a20 + delta_Cl_a20_lef * dlef
        dLdR = (B / (2 * vt)) * (Clr + delta_Clr_lef * dlef)
        dLdP = (B / (2 * vt)) * (Clp + delta_Clp_lef * dlef)
        Cl_tot = Cl + delta_Cl_lef * dlef + dLdail * dail + delta_Cl_r30 * drud + dLdR * R + dLdP * P + delta_Clbeta * beta

        state_dot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        state_old = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ########Compute UVW##########
        Udot = R * V - Q * W - g * sin(pitch * pi / 180) + qbar * S * Cx_tot / m + T / m
        Vdot = P * W - R * U + g * cos(pitch * pi / 180) * sin(roll * pi / 180) + qbar * S * Cy_tot / m
        Wdot = -(Q * U - P * V + g * cos(pitch * pi / 180) * cos(roll * pi / 180) + qbar * S * Cz_tot / m)
        vt_dot = (U * Udot + V * Vdot + W * Wdot) / vt

        v, u, w = pre_state[6], pre_state[7], pre_state[8]
        if (v > 0):
            v += Vdot / 10000.0
        elif (v < 0):
            v -= Vdot / 10000.0
        if (0 <= u < 800.0):
            u += Udot / 10000.0
        elif (-800.0< u < 0):
            u -= Udot / 10000.0
        elif(u >= 800.0):
            u += 0.0
        else:
            u -= 0.0
        if (w > 0):
            w += Wdot / 10000.0
        if (w < 0):
            w -= Wdot / 10000.0
        state_dot[6] = v * 100.0
        state_dot[7] = u * 100.0
        state_dot[8] = w * 100.0
        state_old[6] = v
        state_old[7] = u
        state_old[8] = w
        state_old[18] = math.sqrt(abs(v ** 2) + abs(u ** 2) + abs(w ** 2))

        #########set alpha, beta######
        if (U > 0):
            alpha_dot = -math.atan(W / U) * r2d
        elif (U == 0):
            alpha_dot = 90
        elif (U < 0):
            alpha_dot = math.atan(W / U) * r2d
        beta_dot = math.asin(V / vt) * r2d
        state_dot[12] = alpha_dot
        state_dot[13] = beta_dot
        state_old[12] = alpha_dot
        state_old[13] = beta_dot

        ########Compute PQR##########
        L_tot = Cl_tot * qbar * S * B
        M_tot = Cm_tot * qbar * S * cbar
        N_tot = Cn_tot * qbar * S * B

        denom = Jx * Jz - Jxz * Jxz

        P_tot = (Jz * L_tot + Jxz * N_tot - (Jz * (Jz - Jy) + Jxz * Jxz) * Q * R + Jxz * (
                Jx - Jy + Jz) * P * Q + Jxz * Q * Heng) / denom
        Q_tot = (M_tot + (Jz - Jx) * P * R - Jxz * (P * P - R * R) - R * Heng) / Jy
        R_tot = (Jx * N_tot + Jxz * L_tot + (Jx * (Jx - Jy) + Jxz * Jxz) * P * Q - Jxz * (
                Jx - Jy + Jz) * Q * R + Jx * Q * Heng) / denom
        P += P_tot
        Q += Q_tot
        R += R_tot
        state_dot[9] = P
        state_dot[10] = Q
        state_dot[11] = R
        state_old[9] = P
        state_old[10] = Q
        state_old[11] = R

        ########Euler Angle##########
        if ac_p == 1:
            Q = PLA_LIFT_RATE * dt
        elif ac_p == -1:
            Q = -PLA_LIFT_RATE * dt
        if ac_r == 1:
            P = PLA_ROLL_RATE * dt
        elif ac_r == -1:
            P = -PLA_ROLL_RATE * dt
        if ac_y == 1:
            R = PLA_TURN_RATE * dt
        elif ac_y == -1:
            R = -PLA_TURN_RATE * dt

        heading, pitch, roll = pre_state[3], pre_state[4], pre_state[5]
        phi, theta, psi = roll, pitch, -heading
        sphi = sin(phi * pi / 180)
        cphi = cos(phi * pi / 180)
        st = sin(theta * pi / 180)
        ct = cos(theta * pi / 180)
        tt = tan(theta * pi / 180)
        spsi = sin(psi * pi / 180)
        cpsi = cos(psi * pi / 180)
        if (pitch / 90) % 2 != 1:
            phi += P + tt * (Q * sphi + R * cphi)
            theta += Q * cphi - R * sphi
            psi += (Q * sphi + R * cphi) / ct
        else:
            phi += 0
            theta += Q * cphi - R * sphi
            psi += 0
        heading, pitch, roll = self.__warp(-psi, theta, phi)
        state_dot[3] = heading
        state_dot[4] = pitch
        state_dot[5] = roll
        state_old[3] = heading
        state_old[4] = pitch
        state_old[5] = roll

        ###########new position###############
        V, U, W = pre_state[6], pre_state[7], pre_state[8]
        phi, theta, psi = roll, pitch, -heading
        pi = math.acos(-1)
        sphi = sin(phi * pi / 180)
        cphi = cos(phi * pi / 180)
        st = sin(theta * pi / 180)
        ct = cos(theta * pi / 180)
        spsi = sin(psi * pi / 180)
        cpsi = cos(psi * pi / 180)
        x_dot = U * (ct * spsi) + V * (sphi * spsi * st + cphi * cpsi) + W * (cphi * st * spsi - sphi * cpsi)
        y_dot = U * (ct * cpsi) + V * (sphi * cpsi * st - cphi * spsi) + W * (cphi * st * cpsi + sphi * spsi)
        z_dot = U * st - V * (sphi * ct) - W * (cphi * ct)
        x, y, z = pre_state[1], pre_state[0], pre_state[2]
        state_dot[1] = (x + x_dot * dt) * 100.0
        state_dot[0] = (y + y_dot * dt) * 100.0
        state_dot[2] = (z + z_dot * dt) * 100.0
        state_old[1] = x + x_dot * dt
        state_old[0] = y + y_dot * dt
        state_old[2] = z + z_dot * dt

        if (abs(Q) >= 0.9):
            state_dot[10] = 0.0
            state_old[10] = 0.0
        elif(1e-5 < abs(Q) < 0.9):
            if (Q > 0):
                state_dot[10] -= 5e-5
                state_old[10] -= 5e-5
            else:
                state_dot[10] += 5e-5
                state_old[10] += 5e-5

        if (abs(P) >= 0.9):
            state_dot[9] = 0.0
            state_old[9] = 0.0
        elif(1e-5 < abs(P) < 0.9):
            if (P > 0):
                state_dot[9] -= 5e-5
                state_old[9] -= 5e-5
            else:
                state_dot[9] += 5e-5
                state_old[9] += 5e-5

        if (abs(R) >= 0.01):
            state_dot[11] = 0.0
            state_old[11] = 0.0
        elif(1e-5 < abs(R) < 0.01):
            if (R > 0):
                state_dot[11] -= 5e-5
                state_old[11] -= 5e-5
            else:
                state_dot[11] += 5e-5
                state_old[11] += 5e-5

        state_dot[14] = 5000
        state_old[14] = 5000
        state_dot[15] = 0.0
        state_old[15] = 0.0
        state_dot[16] = 0.0
        state_old[16] = 0.0
        state_dot[17] = 0.0
        state_old[17] = 0.0

        self.old = state_old
        self.sta = state_dot

    # state[0]:x, state[1]:y, state[2]:z, state[3]:yaw, state[7]:v
    # 其余设置均为0,state=[x, y, z, yaw, 0, 0, 0, v, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    def reset(self, state):
        x, y, z = state[1] / 100.0, state[0] / 100.0, state[2] / 100.0
        heading, pitch, roll = state[3], state[4], state[5]
        V, U, W = state[6] / 100.0, state[7] / 100.0, state[8] / 100.0
        vt = math.sqrt(abs(V ** 2) + abs(U ** 2) + abs(W ** 2))
        P, Q, R = state[9], state[10], -state[11]
        alpha, beta = state[12], state[13]
        T, ail, el, rud = state[14], state[15], state[16], state[17]
        self.old = [x, y, z, heading, pitch, roll, V, U, W, P, Q, R, alpha, beta, T, ail, el, rud, vt]
        self.sta = state

    def step(self, dt, action):
        pre_fly = self.old
        self.__dynamics(pre_fly, action, dt)

    def state(self, fighter_dict):
        x, y, z, yaw, pitch, roll, V, U, W, P, Q, R, alpha, beta, T, ail, el, rud = self.sta
        fighter_dict['x'] = x 
        fighter_dict['y'] = y
        fighter_dict['z'] = z
        fighter_dict['v'] = math.sqrt(abs(V ** 2)+abs(U ** 2)+abs(W ** 2))
        fighter_dict['pitch'] = pitch
        fighter_dict['yaw'] = -yaw
        fighter_dict['roll'] = roll
