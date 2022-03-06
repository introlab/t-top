import math

import sympy.vector
from scipy.spatial.transform import Rotation

from state.stewart_forward_kinematics import EULER_ANGLE_SEQUENCE

def main():
    sympy.init_printing(wrap_line=False)

    # Check scipy Rotation and sympy.vector.SpaceOrienter matching
    scipy_rotation_matrix = Rotation.from_euler(EULER_ANGLE_SEQUENCE, [math.pi / 8, math.pi / 16, math.pi / 4]).as_dcm()
    sympy_rotation_matrix = sympy.vector.SpaceOrienter(-math.pi / 8, -math.pi / 16, -math.pi / 4,
                                                       EULER_ANGLE_SEQUENCE).rotation_matrix()

    print('Check scipy Rotation and sympy.vector.SpaceOrienter matching')
    print('scipy')
    print(scipy_rotation_matrix)
    print('sympy')
    sympy.pprint(sympy_rotation_matrix)
    print()

    # Jacobian
    d = sympy.symbols('d')  # Rod length

    bax, bay, baz = sympy.symbols('bax bay baz')  # Bottom anchor
    ba = sympy.Matrix([bax, bay, baz])

    itax, itay, itaz = sympy.symbols('itax, itay, itaz')  # Initial top anchor
    ita = sympy.Matrix([itax, itay, itaz])

    tx, ty, tz = sympy.symbols('tx ty tz')  # Translation
    t = sympy.Matrix([tx, ty, tz])

    rx, ry, rz = sympy.symbols('rx ry rz')  # Rotation
    r = sympy.vector.SpaceOrienter(-rx, -ry, -rz, EULER_ANGLE_SEQUENCE).rotation_matrix()

    ta = r * ita + t
    diff = ba - ta
    error = sympy.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) - d

    print('error =')
    sympy.pprint(error)
    print()

    dtx = error.diff(tx).simplify()
    dty = error.diff(ty).simplify()
    dtz = error.diff(tz).simplify()
    drx = error.diff(rx).simplify()
    dry = error.diff(ry).simplify()
    drz = error.diff(rz).simplify()

    print('dtx =')
    sympy.pprint(dtx)
    print()
    print()
    print('dty =')
    sympy.pprint(dty)
    print()
    print()
    print('dtz =')
    sympy.pprint(dtz)
    print()
    print()
    print('drx =')
    sympy.pprint(drx)
    print()
    print()
    print('dry =')
    sympy.pprint(dry)
    print()
    print()
    print('drz =')
    sympy.pprint(drz)


if __name__ == '__main__':
    main()
