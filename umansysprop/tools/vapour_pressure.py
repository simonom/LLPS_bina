# vim: set et sw=4 sts=4 fileencoding=utf-8:
#
# Copyright (c) 2016 David Topping.
# All Rights Reserved.
# This file is part of umansysprop.
#
# umansysprop is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# umansysprop is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# umansysprop.  If not, see <http://www.gnu.org/licenses/>.

"""
Pure component vapour pressures of organic compounds
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')


from ..forms import (
    Form,
    FloatRangeField,
    SMILESListField,
    SelectField,
    InputRequired,
    NumberRange,
    Length,
    )
from ..results import Result, Table

from .. import boiling_points
from .. import vapour_pressures


class HandlerForm(Form):
    compounds = SMILESListField(
        'Compounds',
        compounds=[
            ('C(CC(=O)O)C(=O)O',                 'Succinic acid'),
            ('C(=O)(C(=O)O)O',                   'Oxalic acid'),
            ('O=C(O)CC(O)=O',                    'Malonic acid'),
            ('CCCCC/C=C/C/C=C/CC/C=C/CCCC(=O)O', 'Pinolenic acid'),
            ])
    temperatures = FloatRangeField(
        'Temperatures (K)', default=298.15, validators=[
            NumberRange(min=173.15, max=400.0, message='Temperatures must be between 173.15K and 400.0K'),
            Length(min=1, max=100, message='Temperatures must have between 1 and 100 values'),
            ])
    vp_method = SelectField(
        'Vapour pressure method', choices=[
            ('nannoolal',             'Nannoolal et al 2008'),
            ('myrdal_and_yalkowsky',  'Myrdal and Yalkowsky 1997'),
            ('evaporation',           'Evaporation - Compernolle et al 2011'),
            ], validators=[InputRequired()])
    bp_method = SelectField(
        'Boiling point method', choices=[
            ('joback_and_reid',  'Joback and Reid 1987'),
            ('stein_and_brown',  'Stein and Brown 1994'),
            ('nannoolal',        'Nannoolal et al 2004'),
            ], validators=[InputRequired()])


def handler(compounds, temperatures, vp_method, bp_method):
    """
    Calculates vapour pressures for all specified *compounds* (given as a
    sequence of SMILES strings) at all given *temperatures* (a sequence of
    floating point values giving temperatures in degrees Kelvin). The
    *vp_method* parameter is one of the strings:

    * 'nannoolal'
    * 'myrdal_and_yalkowsky'
    * 'evaporation'

    Indicating whether to use the methods of Nanoolal et al [NANN2008]_, Myrdal
    and Yalkowsky [MYRDAL1997]_, or evaporation [COMP2011] when
    calculating the vapour
    pressure. Finally, the *bp_method* parameter is one of the strings:

    * 'nannoolal'
    * 'joback_and_reid'
    * 'stein_and_brown'

    Indicating whether to use the methods of Nannoolal et al [NANN2004]_,
    Joback and Reid [JOBACK1987]_, or Stein and Brown [STEIN1994]_ when
    calculating the boiling point for each compound.

    The result is a single table with the temperatures in the rows, and the
    compounds in the columns. Vapour pressures are given as a base 10
    logarithm.

    .. [NANN2004]    Nannoolal, Y., Rarey, J., Ramjugernath, D., and
                     Cordes, W.: Estimation of pure component properties
                     Part 1, Estimation of the normal boiling point of
                     non-electrolyte organic compounds via group
                     contributions and group interactions, Fluid Phase
                     Equilibr., 226, 45–63, 2004
    .. [NANN2008]    Nannoolal, Y., Rarey, J., and Ramjugernath, D.:
                     Estimation of pure component properties. Part 3.
                     Estimation of the vapor pressure of non-electrolyte
                     organic compounds via group contributions and group
                     interactions, Fluid Phase Equilibr., 269, 117–133, 2008
    .. [MYRDAL1997]  Myrdal, P. B. and Yalkowsky, S. H.: Estimating pure
                     component vapor pressures of complex organic molecules,
                     Ind. Eng. Chem. Res., 36, 2494–2499, 1997.
    .. [STEIN1994]   Stein, S. E. and Brown, R. L.: Estimation of normal
                     boiling points from group contributions,
                     J. Chem. Inf. Comp. Sci., 34, 581–587, 1994.
    .. [JOBACK1987]  Joback, K. G. and Reid, R. C.: Estimation of
                     pure-component properties from group-contributions,
                     Chem. Eng. Commun., 57, 233– 243, 1987.
    .. [COMP2011]    Compernolle, S., Ceulemans, K., and Müller, J.-F.:
                     EVAPORATION: a new vapour pressure estimation method
                     for organic molecules including non-additivity and
                     intramolecular interactions, Atmos. Chem. Phys.,
                     11, 9431-9450, doi:10.5194/acp-11-9431-2011, 2011.
    """

    vapour_pressure = {
        'nannoolal':            vapour_pressures.nannoolal,
        'myrdal_and_yalkowsky': vapour_pressures.myrdal_and_yalkowsky,
        # Evaporation doesn't use boiling point
        'evaporation': lambda c, t, b: vapour_pressures.evaporation(c, t),
        }[vp_method]
    boiling_point = {
        'joback_and_reid': boiling_points.joback_and_reid,
        'stein_and_brown': boiling_points.stein_and_brown,
        'nannoolal':       boiling_points.nannoolal,
        }[bp_method]

    data = {}
    for c in compounds:
        b = boiling_point(c)
        for t in temperatures:
            data[(t, str(c).strip())] = vapour_pressure(c, t, b)

    return Result(
        Table(
            'pressures',
            title='Vapour pressure as log₁₀ value [atmospheres]',
            rows_title='Temperature', rows_unit='K', rows=temperatures,
            cols_title='Compound', cols=[str(c).strip() for c in compounds],
            data=data,
            )
        )

