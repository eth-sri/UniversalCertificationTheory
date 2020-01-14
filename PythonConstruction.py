from math import ceil
from numpy import hstack, array, linspace, sqrt, ndarray
import numbers

############################# Intervals ###############################
class Interval(object):
    def __init__(self, l, u):
        assert(l <= u)
        self.l = l
        self.u = u

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.l + other.l, self.u + other.u)
        elif isinstance(other, numbers.Number):
            return Interval(self.l + other, self.u + other)
        else:
            assert(False)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Interval(-self.u, -self.l)

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        assert(not isinstance(other, Interval))
        if other >= 0:
            return Interval(self.l * other, self.u * other)
        elif isinstance(other, numbers.Number):
            return - ((-other) * self)
        else:
            assert(False)

    def __rmul__(self, other):
        return self*other

    def __repr__(self):
        return '['+str(self.l)+','+str(self.u)+']'

################ Basic Building Blocks of the Network #################

def R(x):
    if isinstance(x, numbers.Number):
        return max(x, 0)
    elif isinstance(x, Interval):
        return Interval(R(x.l), R(x.u))
    if isinstance(x, ndarray):
        return array(list(map(R, x)))
    else:
        assert(False)

def R1(x):
    return 1. - R(1.-x)

def nmin2(x, y):
    return 0.5 * (R(x+y) - R(-x-y) - R(x-y) - R(-x+y))

def nmin4(xs):
    assert(len(xs)==4)
    return nmin2(nmin2(xs[0], xs[1]), nmin2(xs[2], xs[3]))

def phi(xs, low_up, M):
    l = 8
    i_low, i_up = low_up
    assert(len(xs)==2)
    ys = M*l*hstack([xs - i_low/M, i_up/M - xs]) + 1
    return R(nmin4(R1(ys)))

############### Function and the associated network ##################

class func_I(object):
    def __init__(self, 
                 func,              # Concrete function
                 Delta_k_func,      # function generating the Delta_k sets
                 delta,             # delta
                 minimum,           # maximum of func on [0,1]^m
                 maximum,           # minimum of func on [0,1]^m
                 lip_const          # Lipschitz constant
                 ):          

        self.func = func
        
        self.M = ceil(2 * lip_const / delta)
        self.delta = 2 * lip_const / self.M # make delta smaller if neccesarry

        self.minimum = minimum

        num_slices = ceil(2 * (maximum - minimum) / delta)
        xis = linspace(minimum, maximum, num_slices+1)
        self.Delta_k_list = [Delta_k_func(xi, self.delta, self.M) for xi in xis]
        
    def __call__(self, x):
        if isinstance(x[0], numbers.Number):
            return self.func(x)
        else:
            assert(False)
    
    def network(self, x):
        res = self.minimum
        for Delta_k in self.Delta_k_list:
            res_k = 0
            for low_up in Delta_k:
                res_k = res_k + phi(x, low_up, self.M)

            res += self.delta / 2 * R1(res_k)
        
        return res

    # Check if Theorem holds on input-box x
    def comp(self, x, func_I_precise):
        precise, approximated = func_I_precise(x), self.network(x)
        ok1 = abs(precise.l - approximated.l) < self.delta + 1e-5
        ok2 = abs(precise.u - approximated.u) < self.delta + 1e-5
        print(str(precise)+'   '+str(approximated)+'   '+str(ok1 and ok2))


##########################################################################
def Delta_k_func(xi_k_1, delta, M, func):
    pts = linspace(0,1,M+1)
    list_of_c = []

    idx, idy = 0, 0
    while idx <= M:
        if func(array([pts[idx], pts[idy]])) >= xi_k_1 + delta/2:
            list_of_c.append(( array([idx, idy]), 
                               array([  M  ,   M  ]) ))
            break
        idx += 1
    
    idx -= 1

    while idy <= M-1 and idx >= 0:
        idy += 1
        if func(array([pts[idx], pts[idy]])) >= xi_k_1 + delta/2:
            while idx >= 0 and func(array([pts[idx], pts[idy]])) >= xi_k_1 + delta/2:
                idx -= 1
            list_of_c.append(( array([(idx+1), idy]), 
                               array([  M  ,   M  ]) ))
            
        
    return list_of_c

######

def square_func(x):
    return x[0]**2 + x[1]**2

def square_func_I_precise(x):
    if x[0].l >= 0 and x[1].l >= 0:
        return Interval(square_func([x[0].l, x[1].l]), square_func([x[0].u, x[1].u]))
    else:
        assert(False)

def square_Delta_k_func(xi_k_1, delta, M):
    return Delta_k_func(xi_k_1, delta, M, square_func)

######

def sqrt_func(x):
    return sqrt(x[0] + x[1] + 1)

def sqrt_func_I_precise(x):
    if x[0].l >= 0 and x[1].l >= 0:
        return Interval(sqrt_func([x[0].l, x[1].l]), sqrt_func([x[0].u, x[1].u]))
    else:
        assert(False)

def sqrt_Delta_k_func(xi_k_1, delta, M):
    return Delta_k_func(xi_k_1, delta, M, sqrt_func)

#########################################################################
delta = 0.1


square_minimum, square_maximum = 0, 2
square_lip_const = 4

square_func_I = func_I(square_func, square_Delta_k_func, 
                       delta,square_minimum, square_maximum,square_lip_const)


sqrt_minimum, sqrt_maximum = 1, sqrt(3)
sqrt_lip_const = 1

sqrt_func_I = func_I(sqrt_func, sqrt_Delta_k_func, delta, 
                     sqrt_minimum, sqrt_maximum, sqrt_lip_const)

########################################################################
###################   x Interval,  y Interval ##########################
comp_list = [
             array([Interval(0,1), Interval(0,1)]),
             array([Interval(0.9,1), Interval(0.9,1)]),
             array([Interval(0.7,1), Interval(0.3,0.4)]),
             array([Interval(0.5,1), Interval(0.2,0.4)]),
             array([Interval(0.2,0.4), Interval(0.5,1)]),
             array([Interval(0.6,0.8), Interval(0.3,0.4)]), 
             array([Interval(1,1), Interval(0.2,0.8)])
            ]

print('Test on 2-dim square function: f(x,y) := x**2 + y**2')
for x_comp in comp_list:
    square_func_I.comp(x_comp, square_func_I_precise)

print('')
print('Test on 2-dim square root function: f(x,y) := sqrt(x+y+1)')
for x_comp in comp_list:
    sqrt_func_I.comp(x_comp, sqrt_func_I_precise)